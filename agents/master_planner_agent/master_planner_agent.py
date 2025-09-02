import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import yaml
from pydantic import ValidationError, BaseModel
from config.agents_io import (
    MasterPlannerInput,
    MasterPlannerOutput,
    TargetFileOutput,
    FileAnalysisResult
)
from utils.llm_client import llm_client
from ast import literal_eval

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MasterPlannerAgent:
    def __init__(self):
        config_path = Path(__file__).parent / "master_planner_config.yaml"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.system_prompt = config.get("system_prompt", "You are a Code Identifier Agent.")
        except Exception as e:
            logger.error(f"[MasterPlannerAgent] Failed to load system prompt: {e}")
            self.system_prompt = "You are a RAG-powered Code Identifier Agent that creates file modification plans based on RAG analysis output."
    async def identify_target_files(self, input_data: MasterPlannerInput, rag_result: str) -> MasterPlannerOutput:
        """
        Identify target files based solely on RAG agent output with comprehensive validation
        
        Args:
            input_data: MasterPlannerInput (without project_path dependency)
            rag_result: Output from your RAG agent as string
        """
        logger.info("🔍 Starting RAG-only file identification process...")
        
        if not rag_result or not rag_result.strip():
            return MasterPlannerOutput(
                success=False,
                message="No RAG analysis result provided.",
                files_to_modify=[]
            )

        try:
            # Extract specific file mentions from user query (still useful for validation)
            specific_files = self._extract_specific_files(input_data.user_question)
            if specific_files:
                logger.info(f"📝 Specific files mentioned by user: {specific_files}")
                
                # Quick check: Are user-specified files even mentioned in RAG?
                missing_from_rag = []
                for file_name in specific_files:
                    if file_name.lower() not in rag_result.lower():
                        missing_from_rag.append(file_name)
                
                if missing_from_rag:
                    return MasterPlannerOutput(
                        success=False,
                        message=f"The user-specified files {missing_from_rag} are not mentioned in the RAG analysis results. "
                            f"The knowledge base may not contain information about these specific files.",
                        files_to_modify=[]
                    )

            # Use RAG output to identify and create target files
            target_files_analysis = await self._analyze_with_rag_only(
                user_query=input_data.user_question,
                rag_output=rag_result,
                config=input_data.parsed_config,
                specific_files=specific_files
            )

            # Check if analysis failed due to insufficient RAG output
            if not target_files_analysis:
                failure_reasons = []
                
                # Determine specific failure reasons
                if not self._has_file_content_in_rag(rag_result):
                    failure_reasons.append("RAG output lacks specific file names or code content")
                
                if specific_files:
                    failure_reasons.append(f"User-specified files {specific_files} could not be properly identified")
                
                if not failure_reasons:
                    failure_reasons.append("RAG analysis output is insufficient for file identification")
                
                failure_message = "File identification failed: " + "; ".join(failure_reasons) + "."
                
                return MasterPlannerOutput(
                    success=False,
                    message=failure_message,
                    files_to_modify=[]
                )

            # Create TargetFileOutput objects from RAG analysis
            target_files = []
            for file_analysis in target_files_analysis:
                try:
                    # Create target file output directly from RAG analysis
                    target_file = self._create_target_file_from_rag(file_analysis)
                    target_files.append(target_file)
                    
                    logger.info(f"✅ Created target file: {target_file.file_path}")
                except Exception as fe:
                    logger.error(f"[FILE ERROR] Failed to create target file from analysis: {fe}")
                    logger.debug(f"File analysis data: {file_analysis}")
                    continue

            if not target_files:
                return MasterPlannerOutput(
                    success=False,
                    message="No valid target files could be created from RAG analysis.",
                    files_to_modify=[]
                )

            # Sort by priority
            priority_order = {"high": 3, "medium": 2, "low": 1}
            target_files.sort(key=lambda x: priority_order.get(x.priority, 0), reverse=True)

            # CRITICAL CHECK: Validate user-specified files are included (fail if missing)
            if specific_files:
                found_file_names = [Path(tf.file_path).name for tf in target_files]
                missing_specific_files = [f for f in specific_files if f not in found_file_names]
                
                if missing_specific_files:
                    logger.error(f"❌ User-specified files not found in final analysis: {missing_specific_files}")
                    return MasterPlannerOutput(
                        success=False,
                        message=f"The user-specified files {missing_specific_files} were not found in the RAG analysis. "
                            f"RAG output may not contain information about these specific files.",
                        files_to_modify=[]
                    )

            logger.info(f"✅ Successfully created {len(target_files)} target files from RAG analysis")
            
            return MasterPlannerOutput(
                success=True,
                message=f"Successfully identified {len(target_files)} files for modification based on RAG analysis.",
                files_to_modify=target_files
            )

        except Exception as e:
            logger.exception(f"❌ Unexpected error during RAG-based file identification: {e}")
            return MasterPlannerOutput(
                success=False,
                message=f"Error processing RAG analysis: {str(e)}",
                files_to_modify=[]
            )
    async def _analyze_with_rag_only(self, user_query: str, rag_output: str, 
                                   config: Dict[str, Any], specific_files: List[str] = None) -> List[Dict[str, Any]]:
        """Create file identification plan using only RAG output"""
        
        prompt = f"""
You are a Code Identifier Agent that creates detailed file modification plans based on RAG (Retrieval-Augmented Generation) analysis output.

USER QUERY: "{user_query}"

RAG ANALYSIS RESULT:
{rag_output}

CONFIGURATION:
{json.dumps(config, indent=2)}

SPECIFIC FILES MENTIONED BY USER: {specific_files or "None"}
TASK:
Based on the RAG analysis output, identify which files need to be created or modified and provide detailed specifications for each file.
INSTRUCTIONS:
1. **Extract File Information**: From the RAG output, identify all files that need to be created or modified
2. ** Migration Query Information** : if its migration please takes all files under consideration irrespective of the file type, Functionality and and requirement make all files be there in the plan.
3. **Identify Cross-File Dependencies** : For each file, determine which other files it depends on or affects
4. **Create File Specifications**: For each file, provide detailed information including:
   - Complete file path (can be relative or absolute as provided by RAG)
   - Detailed analysis of what needs to be modified
   - Priority level based on importance
   - Cross-file dependencies and relationships
3. **Handle Different File Types**:
   - **Existing files to modify**: Extract current structure and modification requirements
   - **New files to create**: Provide complete specifications for creation
   - **Configuration files**: Include any config files mentioned in RAG output
4. **Priority Rules**:
   - Files specifically mentioned by user = HIGH priority
   - Core functionality files = HIGH priority
   - Supporting/utility files = MEDIUM priority
   - Test/documentation files = LOW priority

   ** IMPORTANT Note**:
   - if Their is any change related to migration or upgrate the pyspark version use that as a migration as modification type.
   - U must follow the format of the file_path mentioned below in the response format. Don't use backslash, make sure to use forward slash.
   - Always provide the complete file_path, do not deviate from it.
   - Don't any new file which is not present in rag output strictly.

RESPONSE FORMAT (JSON only):
{{
  "identified_files": [
    {{
      "file_path": "complete/path/to/file.py",
      "file_name": "file.py",
      "priority": "high|medium|low",
      "needs_modification": true,
      "modification_type": "data_loading|data_transformation|configuration|testing|utility|new_file",
      "reason": "detailed explanation from RAG analysis",
      "file_info": {{
        "size": 0,
        "exists": true,
        "extension": ".py",
        "is_python": true
      }},
      "rag_context": "relevant context from RAG output for this file"
      "cross_file_dependencies": {{
        "depends_on": ["path/to/dependency1.py"]
        "affects": ["path/to/affected1.py"]
        "imports_from": ["module1"]
        "imported_by": ["file1.py"]
        "dependency_reason": "explanation of why these dependencies exist"
      }}
    }}
  ],
  "analysis_summary": "summary of the file identification based on RAG output",
  "total_files": 0,
  "confidence_level": "high|medium|low"
}}
**IMPORTANT REQUIREMENTS**:
- Base ALL file identification on the RAG analysis output
- Include comprehensive cross-file dependency analysis
- If RAG mentions specific files, include them with appropriate priority
- If user mentioned specific files, ensure they are included with high priority
- Create complete file specifications even if files don't exist (RAG may suggest new files)
- Include detailed structure information for each file
- Provide comprehensive reasoning based on RAG analysis
- Don't add files not mentioned or implied by RAG analysis
- if RAG mention to include all the files then use all files irrespective of the working and requirement.

** Important Note**:
- For user query related to the migration please include all the files mentioned in the list for the modification.

Return ONLY the JSON response with no additional text.
"""
        try:
            logger.info("🤖 Processing RAG output for file identification...")
            
            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=self.system_prompt
            )
            cleaned_response = self._clean_json_response(response)
            
            try:
                analysis_result = json.loads(cleaned_response)
                identified_files = analysis_result.get('identified_files', [])
                
                logger.info(f"✅ Identified {len(identified_files)} files from RAG analysis")
                logger.info(f"📊 Analysis confidence: {analysis_result.get('confidence_level', 'unknown')}")
                logger.info(f"📋 Summary: {analysis_result.get('analysis_summary', 'No summary')}")
                
                return identified_files
                
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse file identification response: {je}")
                logger.debug(f"Raw response: {response[:500]}...")
                logger.debug(f"Cleaned response: {cleaned_response[:500]}...")
                return []
        except Exception as e:
            logger.error(f"❌ File identification from RAG failed: {e}")
            return []
    def _create_target_file_from_rag(self, file_analysis: Dict[str, Any]) -> TargetFileOutput:
        """Create TargetFileOutput directly from RAG analysis without file system access"""
        
        # Extract file information
        file_path = file_analysis.get('file_path', '')
        file_info = file_analysis.get('file_info', {})
        # structure = file_analysis.get('structure', {})
        
        cross_file_deps = file_analysis.get("cross_file_dependencies", {})
        if cross_file_deps is None:
            cross_file_deps = None
        elif isinstance(cross_file_deps, dict):
            safe_cross_file_deps = {
                 "depends_on": cross_file_deps.get("depends_on",[]) or [],
                 "affects": cross_file_deps.get("affects",[]) or [],
                 "imports_from": cross_file_deps.get("imports_from",[]) or [],
                 "imported_by": cross_file_deps.get("imported_by",[]) or [],
                 "dependency_reason": cross_file_deps.get("dependency_reason",'') or '',
            }

            cross_file_deps = safe_cross_file_deps if any(safe_cross_file_deps.values()) else None
        
        # Create FileAnalysisResult without suggested_changes and cross_file_dependencies
        analysis_result = FileAnalysisResult(
            needs_modification=file_analysis.get('needs_modification', True),
            modification_type=file_analysis.get('modification_type', 'general'),
            priority=file_analysis.get('priority', 'medium'),
            reason=file_analysis.get('reason', 'Identified by RAG analysis'),
            cross_file_dependencies = cross_file_deps
            # Removed: suggested_changes and cross_file_dependencies
        )
        # Create TargetFileOutput
        return TargetFileOutput(
            file_path=file_path,
            file_info=file_info,
            analysis=analysis_result,
            priority=file_analysis.get('priority', 'medium')
        )
    def _extract_specific_files(self, user_question: str) -> List[str]:
        """Extract specific file names mentioned in the user question."""
        file_patterns = [
            r'\b(\w+\.py)\b',
            r'["\']([^"\']*\.py)["\']',
            r'\bin\s+(?:the\s+)?["\']?(\w+\.py)["\']?',
            r'\bmodify\s+(?:the\s+)?["\']?(\w+\.py)["\']?',
            r'\bfile\s+["\']?(\w+\.py)["\']?',
            r'\bcreate\s+(?:the\s+)?["\']?(\w+\.py)["\']?',
            r'\bupdate\s+(?:the\s+)?["\']?(\w+\.py)["\']?',
        ]
        
        specific_files = set()
        for pattern in file_patterns:
            try:
                matches = re.findall(pattern, user_question, re.IGNORECASE)
                if matches:
                    specific_files.update(matches)
            except Exception as e:
                logger.warning(f"Error in pattern matching: {e}")
                continue
        
        # Filter valid files
        valid_files = []
        for file_name in specific_files:
            if (file_name.endswith('.py') and len(file_name) > 3 and 
                re.match(r'^[\w\-\.]+\.py$', file_name)):
                valid_files.append(file_name)
        
        logger.info(f"Extracted specific files: {valid_files}")
        return valid_files
    
    def _has_file_content_in_rag(self, rag_result: str) -> bool:
        return bool(rag_result and len(rag_result.strip()) > 50)
    
    def _clean_json_response(self, response: str) -> str:
        """Clean and extract JSON from LLM response."""
        # Remove markdown code blocks
        json_pattern = r"```(?:json)?(.*?)```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            response = matches[0].strip()
        else:
            # Find JSON object boundaries
            json_start = response.find("{")
            if json_start != -1:
                response = response[json_start:]
                json_end = response.rfind("}")
                if json_end != -1:
                    response = response[:json_end + 1]
        # Clean up common issues
        response = response.replace("True", "true").replace("False", "false")
        response = re.sub(r",\s*([}\]])", r"\1", response)  # Remove trailing commas
        return response.strip()