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
            self.analyze_with_rag_prompt = config.get("analyze_with_rag_prompt")
            self.detect_migration_prompt = config.get("detect_migration_prompt")
            self.detect_migration_type_prompt = config.get("detect_migration_type_prompt")
        except Exception as e:
            logger.error(f"[MasterPlannerAgent] Failed to load system prompt: {e}")
            self.system_prompt = "You are a RAG-powered Code Identifier Agent that creates file modification plans based on RAG analysis output."
            self.analyze_with_rag_prompt = "You need to analyze rag_input and create target_files"
            self.detect_migration_prompt = "You have to detect whether this is a migration request or not"
            self.detect_migration_type_prompt = "You have to detect the type of migration"

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

        prompt = self.analyze_with_rag_prompt.format(user_query=user_query, rag_output = rag_output, config=json.dumps(config, indent=2), specific_files=specific_files)

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
        print("I am printing the modification_type from master planner")
        print(file_analysis.get('modification_type', 'utility'))
        analysis_result = FileAnalysisResult(
            needs_modification=file_analysis.get('needs_modification', True),
            modification_type=file_analysis.get('modification_type', 'utility').lower(),
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

    async def detect_migration_with_llm(self,user_query: str) -> Dict[str, Any] :
        detection_prompt = self.detect_migration_prompt.format(user_query=user_query)
        try:
            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": detection_prompt}],
                system_prompt="you are an expert at analyzing code modification requests. Be precise in distinguishing migration vs modification requests",
                temperature=0.1
            )
            raw_content =  response.strip()
            if raw_content.startswith(("'''","```")):
                lines = raw_content.splitlines()
                raw_content = "\n".join(lines[1:-1]).strip()

            detection_result = json.loads(raw_content)

            if detection_result.get("confidence", 0) >= 0.8:
                return detection_result

            return {"is_migration" :False , "confidence" : detection_result.get("confidence", 0.0)}

        except Exception as e:
            print(f"LLM migration detection failed : {e}")
            return {"is_migration" : False, "confidence" : 0.0}

    async def extract_repos_from_query(self,query:str) -> list:

        repo_list = ["examples"]

        DEFAULT_REPO = "examples"

        system_prompt = (
            "Extract all repository names from the following user query."
            "Return only a valid Python list of strings. Example: ['repo1','repo2']"
        )

        try:
            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": query}],
                system_prompt=system_prompt,
                temperature=0.1
            )
            extracted_list = literal_eval(response) if response.startswith("[") else []

            print(f"Extracted Repos : {extracted_list}")

            matched_repos = [
                r for r in extracted_list
                if any(r.lower() == valid.lower() for valid in repo_list)
            ]

            return matched_repos if matched_repos else [DEFAULT_REPO]

        except Exception as e :
            print(f"LLM Extraction failed: {e}")
            return [DEFAULT_REPO]

    async def detect_migration_type_with_llm(self, user_query: str) -> bool:
        prompt = self.detect_migration_type_prompt.format(user_query=user_query)
        try:
            raw_response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )

            possible_types = [
                "single_file_migration",
                "multiple_files_migration"
            ]

            migration_type = raw_response.strip().lower()
            if migration_type[0] == '"' and migration_type[-1] == '"':
                migration_type = migration_type[1:-1]

            if migration_type == "repository_migration":
                return True
            else:
                return False

        except Exception as e:
            return True






