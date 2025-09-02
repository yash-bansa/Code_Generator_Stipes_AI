import json
import re
import logging
from typing import List, Dict, Any, Union
from pathlib import Path
import yaml
from utils.file_handler import FileHandler
from utils.llm_client import llm_client
from config.agents_io import DeltaAnalyzerInput, DeltaAnalyzerOutput

logger = logging.getLogger(__name__)

class DeltaAnalyzerAgent:
    def __init__(self):
        config_path = Path(__file__).parent / "delta_analyzer_config.yaml"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.system_prompt = config["system_prompt"]
        except Exception as e:
            logger.error(f"[delta_analyzer_agent] Failed to load config: {e}")
            self.system_prompt = "You are a delta analyzer agent. (default fallback prompt)"

    async def suggest_file_changes(self, target_file: Dict[str, Any], parsed_config: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
        file_path = target_file.get('file_path')
        if not file_path:
            logger.warning("[DeltaAnalyzerAgent] Missing file_path in target_file")
            return {"modifications": []}
        
        # Handle both full paths and filenames
        file_content = ""
        try:
            if file_path:
                working_file_path = Path.cwd() / file_path
                print("in if")
                logger.info(f"[DeltaAnalyzerAgent] Using filename only: {file_path}")
                file_content = FileHandler.read_file(working_file_path)
            else:
                print("in else")
                working_file_path = Path(file_path)
                file_content = FileHandler.read_file(working_file_path)
        except Exception as e:
            logger.warning(f"[DeltaAnalyzerAgent] Could not read file {file_path}: {e}")
            file_content = f"# could not read file {file_path} : {e}"

        # Enhanced input extraction with safe structure handling
        file_analysis = target_file.get('analysis', {})
        file_info = target_file.get('file_info', {})

        
        # Create DeltaAnalyzerInput with user query
        input_data = DeltaAnalyzerInput(
            file_path=file_path,
            file_content=file_content,
            file_analysis=file_analysis,
            config=parsed_config,
            user_query=user_query
        )

        # Enhanced prompt with user query context
        prompt = self._build_comprehensive_prompt(
            input_data, file_info, file_analysis, user_query
        )

        try:
            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=self.system_prompt
            )
            if response:
                cleaned_response = self._clean_json_response(response)
                parsed = DeltaAnalyzerOutput.model_validate_json(cleaned_response)
                
                # Add user query analysis
                query_analysis = self._analyze_user_query(user_query)
                alignment_score = self._calculate_alignment_score(parsed.dict(), user_query)
                
                return {
                    **parsed.dict(),
                    "file_path": file_path,
                    "original_file_content" : file_content,
                    "timestamp": file_info.get('modified', 0),
                    "original_priority": file_analysis.get('priority', 'medium'),
                    "modification_type": file_analysis.get('modification_type', 'general'),
                    "user_query_analysis": query_analysis,
                    "alignment_score": alignment_score
                }
        except json.JSONDecodeError as je:
            logger.warning(f"[DeltaAnalyzerAgent] JSON decode error for {file_path}: {je}")
        except Exception as e:
            logger.error(f"[DeltaAnalyzerAgent] Error analyzing file {file_path}: {e}")
        
        return {"modifications": [],
                "original_file_content" : file_content}

    def _build_comprehensive_prompt(self, input_data: DeltaAnalyzerInput,
                                  file_info: Dict[str, Any],
                                  file_analysis: Dict[str, Any],
                                  user_query: str = "") -> str:
        """Build enhanced prompt with user query context and safe structure handling"""
        
        # Extract analysis information
        modification_type = file_analysis.get('modification_type', 'general')
        reason = file_analysis.get('reason', '')
        
        # Enhanced prompt with user query context
        prompt = f"""
Generate detailed code modification suggestions for this file based on the user's specific request.

USER QUERY: "{user_query}"

FILE INFORMATION:
- Path: {input_data.file_path}
- Size: {file_info.get('size', 'unknown')} bytes
- Extension: {file_info.get('extension', 'unknown')}
- Modification Type: {modification_type}
- Priority: {file_analysis.get('priority', 'medium')}
- Needs Modification: {file_analysis.get('needs_modification', True)}

ANALYSIS INSIGHTS:
- Modification Reason: {reason}
- RAG Context: {file_analysis.get('rag_context', 'No additional context')}

CONFIGURATION REQUIREMENTS:
{json.dumps(input_data.config, indent=2)}

CURRENT FILE CONTENT:
{input_data.file_content}

INSTRUCTIONS:
1. **PRIORITIZE USER INTENT**: Focus on exactly what the user requested in their query: "{user_query}"
2. **ALIGN WITH USER GOALS**: Ensure all modifications directly support the user's stated objectives
3. **USER-CENTRIC IMPLEMENTATION**: Create implementations that match the user's requirements
4. If file content appears to be mock/placeholder, focus on creating actual implementation that fulfills user requirements
5. Use existing code structure for precise line numbers (handle cases where structure might be empty)
6. Ensure compatibility with identified functions, classes, and variables
7. Address the specific modification type: {modification_type}
8. **CONTEXT-AWARE CHANGES**: Make changes that make sense in the context of the user's overall request

USER QUERY ANALYSIS:
- Extract key technical requirements from: "{user_query}"
- Identify specific technologies, frameworks, or patterns mentioned
- Understand the scope and scale of changes requested
- Consider any constraints or preferences implied in the query

EXPECTED JSON FORMAT:
{{
  "modifications": [
    {{
      "action": "add|modify|delete",
      "target_type": "function|class|import|variable|comment",
      "target_name": "name of the target element",
      "line_number": 0,
      "old_code": "existing code (if applicable)",
      "new_code": "new or modified code that directly addresses user query",
      "explanation": "detailed explanation linking this change to the user's specific request",
      "affects_dependencies": ["list of files that might be affected"],
      "user_intent_alignment": 0.9
    }}
  ],
  "new_dependencies": ["list of new imports/packages needed to fulfill user request"],
  "testing_suggestions": ["testing approaches specific to user's requirements"],
  "potential_issues": ["issues considering user's request and existing structure"],
  "cross_file_impacts": ["impacts on other files based on user's overall request"],
  "implementation_notes": ["guidance for implementing changes that meet user expectations"],
  "user_query_analysis": {{
    "query_type": "migration|feature_addition|bug_fix|optimization",
    "technologies_mentioned": ["list of technologies"],
    "complexity_estimate": "low|medium|high"
  }},
  "alignment_score": 0.85
}}

Return ONLY a valid JSON object with detailed code changes that directly address the user's query.
"""
        return prompt

    async def create_modification_plan(self, target_files: List[Union[Dict[str, Any], Any]], parsed_config: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
        plan = {
            "files_to_modify": [],
            "execution_order": [],
            "dependencies": [],
            "estimated_complexity": "low",
            "risks": [],
            "backup_required": True,
            "cross_file_impact": [],
            "user_query_alignment": {},
            "overall_alignment_score": 0.0
        }

        # Analyze user query for better planning
        query_analysis = self._analyze_user_query(user_query)
        plan["user_query_alignment"] = query_analysis

        # Convert and process each target file
        converted_files = []
        for target_file in target_files:
            file_dict = self._convert_to_dict(target_file)
            converted_files.append(file_dict)

        # Process each converted file with user query context
        alignment_scores = []
        for file_dict in converted_files:
            # Pass user query to file analysis
            suggestions = await self.suggest_file_changes(file_dict, parsed_config, user_query)
            
            file_analysis = file_dict.get('analysis', {})
            actual_priority = file_analysis.get('priority', 'medium')
            
            # Calculate alignment score for this file
            file_alignment_score = suggestions.get('alignment_score', 0.5)
            alignment_scores.append(file_alignment_score)
            
            plan["files_to_modify"].append({
                "file_path": file_dict.get('file_path', 'unknown'),
                "priority": actual_priority,
                "modification_type": file_analysis.get('modification_type', 'general'),
                "suggestions": suggestions,
                "user_alignment_score": file_alignment_score,
                "basic_info": {
                    "needs_modification": file_analysis.get('needs_modification', True),
                    "reason": file_analysis.get('reason', 'Modification needed')
                }
            })

        # Calculate overall alignment score
        plan["overall_alignment_score"] = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

        # Enhanced execution order with user query priorities
        plan["execution_order"] = self._determine_user_aware_execution_order(
            plan["files_to_modify"], user_query
        )
        
        # Enhanced planning components
        plan["estimated_complexity"] = self._calculate_enhanced_complexity(plan["files_to_modify"])
        plan["risks"] = self._assess_risks_with_user_context(plan["files_to_modify"], user_query)
        plan["dependencies"] = self._extract_basic_dependencies(converted_files)
        plan["cross_file_impact"] = self._analyze_basic_cross_file_impact(converted_files)

        return plan

    def _convert_to_dict(self, target_file: Any) -> Dict[str, Any]:
        """Convert TargetFileOutput (Pydantic) to dictionary with safe structure handling"""
        
        if isinstance(target_file, dict):
            # Already a dictionary
            file_dict = target_file
        elif hasattr(target_file, 'dict'):
            # Pydantic v1 style
            file_dict = target_file.dict()
        elif hasattr(target_file, 'model_dump'):
            # Pydantic v2 style
            file_dict = target_file.model_dump()
        elif hasattr(target_file, '__dict__'):
            # Regular object
            file_dict = target_file.__dict__
        else:
            # Fallback - create basic structure
            file_dict = {
                "file_path": getattr(target_file, 'file_path', 'unknown'),
                "file_info": getattr(target_file, 'file_info', {}),
                "analysis": getattr(target_file, 'analysis', {}),
                "priority": getattr(target_file, 'priority', 'medium')
            }
        
       
        # Ensure analysis is also converted if it's a Pydantic object
        analysis = file_dict.get('analysis')
        if analysis and hasattr(analysis, 'dict'):
            file_dict['analysis'] = analysis.dict()
        elif analysis and hasattr(analysis, 'model_dump'):
            file_dict['analysis'] = analysis.model_dump()
        elif analysis is None:
            file_dict['analysis'] = {}
        
        return file_dict

    def _analyze_user_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query to extract key requirements and priorities"""
        
        analysis = {
            "query_type": "general",
            "technologies_mentioned": [],
            "priority_keywords": [],
            "scope_indicators": [],
            "complexity_indicators": [],
            "action_words": []
        }
        
        if not user_query:
            return analysis
        
        query_lower = user_query.lower()
        
        # Detect query type
        if any(word in query_lower for word in ["migrate", "convert", "replace", "upgrade"]):
            analysis["query_type"] = "migration"
        elif any(word in query_lower for word in ["add", "implement", "create", "build"]):
            analysis["query_type"] = "feature_addition"
        elif any(word in query_lower for word in ["fix", "bug", "error", "issue"]):
            analysis["query_type"] = "bug_fix"
        elif any(word in query_lower for word in ["optimize", "improve", "performance"]):
            analysis["query_type"] = "optimization"
        
        # Extract technologies
        tech_keywords = ["pandas", "pyspark", "numpy", "tensorflow", "pytorch", "flask", "django", "fastapi", "react", "vue", "angular", "spark", "hadoop"]
        analysis["technologies_mentioned"] = [tech for tech in tech_keywords if tech in query_lower]
        
        # Extract priority indicators
        priority_keywords = ["urgent", "critical", "important", "asap", "quickly", "slowly", "carefully"]
        analysis["priority_keywords"] = [word for word in priority_keywords if word in query_lower]
        
        # Extract scope indicators
        scope_keywords = ["all files", "entire project", "specific file", "only", "just", "complete", "partial"]
        analysis["scope_indicators"] = [phrase for phrase in scope_keywords if phrase in query_lower]
        
        # Extract action words
        action_words = ["migrate", "convert", "add", "remove", "update", "modify", "create", "delete"]
        analysis["action_words"] = [word for word in action_words if word in query_lower]
        
        return analysis
    

    def _calculate_alignment_score(self, suggestions: Dict[str, Any], user_query: str) -> float:
        """Calculate how well the suggestions align with user query"""
        
        if not user_query or not suggestions:
            return 0.5
        
        score = 0.0
        total_checks = 0
        
        query_lower = user_query.lower()
        modifications = suggestions.get('modifications', [])
        
        # Check if modifications mention technologies from user query
        for tech in ["pandas", "pyspark", "numpy", "flask", "django", "spark"]:
            if tech in query_lower:
                total_checks += 1
                if any(tech in mod.get('new_code', '').lower() or tech in mod.get('explanation', '').lower() 
                      for mod in modifications):
                    score += 1.0
        
        # Check if new dependencies align with user request
        new_deps = suggestions.get('new_dependencies', [])
        if new_deps:
            total_checks += 1
            if any(tech in ' '.join(new_deps).lower() for tech in ["pyspark", "pandas", "numpy"] if tech in query_lower):
                score += 1.0
        
        # Check if explanations reference user intent
        total_checks += 1
        if any('user' in mod.get('explanation', '').lower() or 'request' in mod.get('explanation', '').lower() 
              for mod in modifications):
            score += 0.5
        
        # Check for action alignment
        action_words = ["migrate", "convert", "add", "remove", "update"]
        for action in action_words:
            if action in query_lower:
                total_checks += 1
                if any(action in mod.get('explanation', '').lower() for mod in modifications):
                    score += 0.8
        
        return min(score / total_checks if total_checks > 0 else 0.5, 1.0)

    def _determine_user_aware_execution_order(self, files_to_modify: List[Dict[str, Any]], user_query: str) -> List[str]:
        """Determine execution order considering user query priorities"""
        
        # Start with priority-based ordering
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_files = sorted(files_to_modify, 
                            key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))
        
        # Boost files that align well with user query
        sorted_files.sort(key=lambda x: x.get('user_alignment_score', 0.5), reverse=True)
        
        order = []
        
        # Apply keyword-based ordering with user query awareness
        priority_keywords = ['config', 'util', 'load', 'input', 'process', 'transform', 'output', 'write']
        
        # Extract user-specific keywords
        if user_query:
            query_keywords = [word.lower() for word in user_query.split() if len(word) > 3]
            priority_keywords = query_keywords[:3] + priority_keywords  # Add top 3 query keywords
        
        for priority in ['high', 'medium', 'low']:
            priority_files = [f for f in sorted_files if f.get('priority') == priority]
            
            # Apply keyword ordering
            for keyword in priority_keywords:
                for file_info in priority_files:
                    file_path = Path(file_info["file_path"])
                    if keyword in file_path.name.lower() and file_info["file_path"] not in order:
                        order.append(file_info["file_path"])
            
            # Add remaining files
            for file_info in priority_files:
                if file_info["file_path"] not in order:
                    order.append(file_info["file_path"])

        return order

    def _assess_risks_with_user_context(self, files_to_modify: List[Dict[str, Any]], user_query: str) -> List[str]:
        """Assess risks considering user query context"""
        
        risks = []
        
        # Standard risk assessment
        high_priority_count = sum(1 for f in files_to_modify if f.get("priority") == "high")
        if high_priority_count > 2:
            risks.append("Multiple high-priority files require modification")
        
        total_modifications = sum(len(f["suggestions"].get("modifications", [])) for f in files_to_modify)
        if total_modifications > 15:
            risks.append("Large number of modifications detected")
        
        # User query specific risks
        if user_query:
            query_lower = user_query.lower()
            
            if "migrate" in query_lower or "convert" in query_lower:
                risks.append("Migration project detected - thorough testing recommended")
            
            if any(tech in query_lower for tech in ["pyspark", "tensorflow", "pytorch"]):
                risks.append("Big data/ML framework integration - performance testing needed")
            
            if "entire project" in query_lower or "all files" in query_lower:
                risks.append("Project-wide changes requested - staged rollout recommended")
            
            if "performance" in query_lower or "optimize" in query_lower:
                risks.append("Performance optimization requested - benchmark testing required")
        
        # Check alignment scores for additional risks
        low_alignment_files = [f for f in files_to_modify if f.get('user_alignment_score', 0.5) < 0.3]
        if low_alignment_files:
            risks.append(f"Low user query alignment detected in {len(low_alignment_files)} files")
        
        return risks

    def _extract_basic_dependencies(self, target_files: List[Dict[str, Any]]) -> List[str]:
        """Extract basic file dependencies (simplified)."""
        dependencies = []
        
        for target_file in target_files:
            file_path = target_file.get('file_path', '')
            file_name = Path(file_path).name if file_path else 'unknown'
            
            # Basic dependency inference from file names and types
            if 'config' in file_name.lower():
                dependencies.append(f"{file_name} - configuration dependency")
            elif 'util' in file_name.lower():
                dependencies.append(f"{file_name} - utility dependency")
            elif 'main' in file_name.lower():
                dependencies.append(f"{file_name} - main execution dependency")
            elif 'load' in file_name.lower():
                dependencies.append(f"{file_name} - data loading dependency")
        
        return dependencies

    def _calculate_enhanced_complexity(self, files_to_modify: List[Dict[str, Any]]) -> str:
        """Enhanced complexity calculation considering multiple factors."""
        total_modifications = sum(len(f["suggestions"].get("modifications", [])) for f in files_to_modify)
        high_priority_files = sum(1 for f in files_to_modify if f.get("priority") == "high")
        avg_alignment = sum(f.get("user_alignment_score", 0.5) for f in files_to_modify) / len(files_to_modify) if files_to_modify else 0.5
        
        # Complexity scoring with alignment consideration
        complexity_score = total_modifications + (high_priority_files * 5)
        
        # Reduce complexity if high alignment (well-understood requirements)
        if avg_alignment > 0.8:
            complexity_score *= 0.8
        elif avg_alignment < 0.3:
            complexity_score *= 1.3  # Increase complexity for poorly aligned requirements
        
        if complexity_score > 25:
            return "high"
        elif complexity_score > 10:
            return "medium"
        else:
            return "low"

    def _analyze_basic_cross_file_impact(self, target_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze basic cross-file impact (simplified)."""
        impacts = []
        
        for target_file in target_files:
            file_path = target_file.get('file_path', '')
            file_name = Path(file_path).name if file_path else 'unknown'
            
            # Basic impact analysis
            impact_info = {
                "source_file": file_path,
                "file_type": self._classify_file_type(file_name),
                "potential_impact": self._assess_file_impact(file_name)
            }
            
            impacts.append(impact_info)
        
        return impacts

    def _classify_file_type(self, file_name: str) -> str:
        """Classify file type based on name."""
        name_lower = file_name.lower()
        
        if 'config' in name_lower or 'setting' in name_lower:
            return "configuration"
        elif 'util' in name_lower or 'helper' in name_lower:
            return "utility"
        elif 'main' in name_lower or 'app' in name_lower:
            return "main_application"
        elif 'test' in name_lower:
            return "test"
        elif 'data' in name_lower or 'load' in name_lower:
            return "data_processing"
        else:
            return "general"

    def _assess_file_impact(self, file_name: str) -> str:
        """Assess potential impact of modifying this file."""
        file_type = self._classify_file_type(file_name)
        
        impact_map = {
            "configuration": "High - affects entire application",
            "utility": "Medium - affects dependent modules",
            "main_application": "High - affects application execution",
            "test": "Low - affects testing only",
            "data_processing": "Medium - affects data flow",
            "general": "Medium - standard impact"
        }
        
        return impact_map.get(file_type, "Medium - standard impact")

    def _clean_json_response(self, response: str) -> str:
        """Extract and clean JSON-like content from LLM response."""
        json_pattern = r"```(?:json)?(.*?)```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            response = matches[0].strip()
        
        response = response.replace('\n', '')
        response = response.replace("True", "true").replace("False", "false")
        response = re.sub(r",\s*([}\]])", r"\1", response)
        return response.strip()