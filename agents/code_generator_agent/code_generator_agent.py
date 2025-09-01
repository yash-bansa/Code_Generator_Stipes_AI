import json
import re
import logging
import yaml
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from utils.llm_client import llm_client
from config.agents_io import (
    CodeGeneratorInput, 
    CodeGeneratorOutput, 
    GeneratedFile, 
    FileModificationPlan, 
    ModificationItem
)

logger = logging.getLogger(__name__)

class CodeGeneratorAgent:
    def __init__(self):
        config_path = Path(__file__).parent / "code_generator_config.yaml"
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.system_prompt = self.config["system_prompt"]
            self.modification_prompt = self.config["modification_prompt"]
        except Exception as e:
            logger.error(f"[CodeGeneratorAgent] Failed to load config: {e}")
            self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration as fallback"""
        self.system_prompt = "You are a Code Generator Agent. Generate clean, well-documented Python code."
        self.modification_prompt = """Apply the following modifications to this code:
{modifications_json}
Current code:
{current_content}
Return ONLY the complete modified Python file code."""
    
    async def generate_code_modifications(self, input_data: CodeGeneratorInput) -> CodeGeneratorOutput:
        """Main method to generate code modifications based on DeltaAnalyzer plan"""
        start_time = time.time()
        
        logger.info(f"[CodeGeneratorAgent] Starting code generation for {len(input_data.modification_plan.get('files_to_modify', []))} files")
        
        output = CodeGeneratorOutput(
            success=True,
            modified_files=[],
            failed_files=[],
            errors=[],
            warnings=[],
            total_modifications=0
        )
        
        modification_plan = input_data.modification_plan
        files_to_modify = modification_plan.get("files_to_modify", [])
        
        for file_info in files_to_modify:
            try:
                result = await self._process_single_file(file_info, input_data)
                if result:
                    output.modified_files.append(result)
                    output.total_modifications += result.modifications_applied
                else:
                    file_path = file_info.get("file_path", "unknown")
                    output.failed_files.append(file_path)
                    output.errors.append(f"Failed to process file: {file_path}")
                    
            except Exception as e:
                file_path = file_info.get("file_path", "unknown")
                logger.exception(f"[CodeGeneratorAgent] Error processing file {file_path}")
                output.failed_files.append(file_path)
                output.errors.append(f"Error processing {file_path}: {str(e)}")
        
        # Determine overall success
        output.success = len(output.failed_files) == 0
        output.execution_time = time.time() - start_time
        
        logger.info(f"[CodeGeneratorAgent] Completed: {len(output.modified_files)} successful, {len(output.failed_files)} failed")
        
        return output
    
    async def _process_single_file(self, file_info: Dict[str, Any], input_data: CodeGeneratorInput) -> Optional[GeneratedFile]:
        """Process a single file modification using content from DeltaAnalyzer output"""
        file_path = file_info.get("file_path")
        if not file_path:
            logger.warning("[CodeGeneratorAgent] Missing file_path in file_info")
            return None
        
        suggestions = file_info.get("suggestions", {})
        
        # Use the original file content from DeltaAnalyzer (no file read needed!)
        current_content = suggestions.get("original_file_content", "")
        
        if not current_content:
            logger.error(f"[CodeGeneratorAgent] No original_file_content provided by DeltaAnalyzer for {file_path}")
            return None
        
        # Extract modifications from DeltaAnalyzer suggestions
        modifications = suggestions.get("modifications", [])
        
        if not modifications:
            logger.info(f"[CodeGeneratorAgent] No modifications specified for {file_path}")
            return None
        
        # Try different modification approaches
        modified_content = None
        
        # First attempt: Direct code replacement using old_code/new_code pairs
        if self._can_use_direct_replacement(modifications):
            logger.info(f"[CodeGeneratorAgent] Using direct replacement for {file_path}")
            modified_content = await self._apply_direct_replacements(
                current_content=current_content,
                modifications=modifications,
                new_dependencies=suggestions.get("new_dependencies", []),
                file_path=file_path
            )
        
        # Second attempt: LLM-based modification if direct replacement fails
        if not modified_content:
            logger.info(f"[CodeGeneratorAgent] Using LLM-based modification for {file_path}")
            modified_content = await self._apply_llm_modifications(
                current_content=current_content,
                modifications=modifications,
                new_dependencies=suggestions.get("new_dependencies", []),
                file_path=file_path,
                user_query=input_data.user_query,
                suggestions=suggestions
            )
        
        if not modified_content:
            logger.error(f"[CodeGeneratorAgent] Failed to generate modifications for {file_path}")
            return None
        
        # *** REMOVED FILE WRITING SECTION ***
        # Code is now kept in memory only
        logger.info(f"[CodeGeneratorAgent] Successfully generated modified content for {file_path}")
        
        return GeneratedFile(
            file_path=file_path,
            original_content=current_content,
            modified_content=modified_content,
            modifications_applied=len(modifications),
            backup_path=None
        )
    
    def _can_use_direct_replacement(self, modifications: List[Dict[str, Any]]) -> bool:
        """Check if we can use direct string replacement for all modifications"""
        for mod in modifications:
            action = mod.get('action', '')
            old_code = mod.get('old_code', '')
            
            # Direct replacement works best for simple modify/delete actions with clear old_code
            if action in ['modify', 'delete'] and not old_code:
                return False
            
            # Complex additions might need LLM understanding
            if action == 'add' and len(mod.get('new_code', '')) > 500:  # Arbitrary threshold
                return False
        
        return True
    
    async def _apply_direct_replacements(self, current_content: str, modifications: List[Dict[str, Any]], 
                                       new_dependencies: List[str], file_path: str) -> Optional[str]:
        """Apply code modifications using direct old_code -> new_code replacements"""
        
        try:
            modified_content = current_content
            
            # Sort modifications by line number (descending) to avoid offset issues
            sorted_modifications = sorted(
                [m for m in modifications if m.get('line_number', 0) > 0], 
                key=lambda x: x.get('line_number', 0), 
                reverse=True
            )
            
            # Add modifications without line numbers at the end
            sorted_modifications.extend([m for m in modifications if m.get('line_number', 0) <= 0])
            
            # Process each modification
            for mod in sorted_modifications:
                action = mod.get('action', '')
                old_code = mod.get('old_code', '').strip()
                new_code = mod.get('new_code', '').strip()
                target_name = mod.get('target_name', '')
                
                if action == 'add':
                    # Add new code at specified location or appropriate place
                    modified_content = self._add_code_segment(
                        modified_content, new_code, mod.get('line_number', -1), target_name
                    )
                    logger.info(f"[CodeGeneratorAgent] Added code segment '{target_name}' in {file_path}")
                
                elif action == 'modify':
                    # Replace old_code with new_code
                    if old_code and new_code:
                        if old_code in modified_content:
                            modified_content = modified_content.replace(old_code, new_code, 1)  # Replace only first occurrence
                            logger.info(f"[CodeGeneratorAgent] Modified code segment '{target_name}' in {file_path}")
                        else:
                            logger.warning(f"[CodeGeneratorAgent] Old code not found for modification '{target_name}' in {file_path}")
                            return None  # Fall back to LLM
                
                elif action == 'delete':
                    # Remove old_code
                    if old_code:
                        if old_code in modified_content:
                            modified_content = modified_content.replace(old_code, '', 1)  # Remove only first occurrence
                            logger.info(f"[CodeGeneratorAgent] Deleted code segment '{target_name}' in {file_path}")
                        else:
                            logger.warning(f"[CodeGeneratorAgent] Old code not found for deletion '{target_name}' in {file_path}")
                            return None  # Fall back to LLM
            
            # Add new dependencies at the top
            if new_dependencies:
                modified_content = self._add_imports(modified_content, new_dependencies)
            
            return modified_content
            
        except Exception as e:
            logger.error(f"[CodeGeneratorAgent] Direct replacement failed for {file_path}: {e}")
            return None
    
    async def _apply_llm_modifications(self, current_content: str, modifications: List[Dict[str, Any]], 
                                     new_dependencies: List[str], file_path: str, user_query: str,
                                     suggestions: Dict[str, Any]) -> Optional[str]:
        """Apply modifications using LLM with enhanced context from DeltaAnalyzer"""
        
        modifications_json = json.dumps(modifications, indent=2)
        
        # Enhanced prompt using DeltaAnalyzer insights
        prompt = self.modification_prompt.format(
            user_query=user_query,
            file_path=file_path,
            current_content=current_content,
            modifications_json=modifications_json,
            new_dependencies=json.dumps(new_dependencies)
        )
        
        # Add additional context from suggestions if available
        if suggestions:
            context_info = f"""
ADDITIONAL CONTEXT:
- Alignment Score: {suggestions.get('alignment_score', 'N/A')}
- Testing Suggestions: {json.dumps(suggestions.get('testing_suggestions', []), indent=2)}
- Implementation Notes: {json.dumps(suggestions.get('implementation_notes', []), indent=2)}
- Potential Issues: {json.dumps(suggestions.get('potential_issues', []), indent=2)}
"""
            prompt += context_info
        
        try:
            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=self.system_prompt,
                temperature=0.1
            )
            
            if response:
                # Clean the response to extract just the code
                cleaned_code = self._extract_code_from_response(response)
                if cleaned_code and len(cleaned_code.strip()) > 0:
                    return cleaned_code
            
            logger.warning(f"[CodeGeneratorAgent] Empty response from LLM for {file_path}")
            return None
                
        except Exception as e:
            logger.error(f"[CodeGeneratorAgent] LLM call failed for {file_path}: {e}")
            return None
    
    def _add_code_segment(self, content: str, new_code: str, line_number: int, target_name: str = "") -> str:
        """Add new code at specified line number or appropriate location"""
        lines = content.split('\n')
        
        if line_number > 0 and line_number <= len(lines):
            # Insert at specific line number
            lines.insert(line_number - 1, new_code)
        else:
            # Smart insertion based on target type
            if target_name:
                insert_index = self._find_best_insertion_point(lines, target_name, new_code)
                lines.insert(insert_index, new_code)
            else:
                # Append at end
                lines.append(new_code)
        
        return '\n'.join(lines)
    
    def _find_best_insertion_point(self, lines: List[str], target_name: str, new_code: str) -> int:
        """Find the best place to insert new code based on context"""
        
        # If it's an import, add with other imports
        if new_code.strip().startswith(('import ', 'from ')):
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith(('import ', 'from ', '#')):
                    return i
            return 0
        
        # If it's a class, add after other classes or at end
        if new_code.strip().startswith('class '):
            last_class_end = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('class '):
                    # Find end of this class
                    j = i + 1
                    while j < len(lines) and (lines[j].startswith('    ') or lines[j].strip() == ''):
                        j += 1
                    last_class_end = j
            
            if last_class_end > 0:
                return last_class_end
        
        # If it's a function, add after other functions or at end
        if new_code.strip().startswith(('def ', 'async def ')):
            last_function_end = -1
            for i, line in enumerate(lines):
                if line.strip().startswith(('def ', 'async def ')):
                    # Find end of this function
                    j = i + 1
                    while j < len(lines) and (lines[j].startswith('    ') or lines[j].strip() == ''):
                        j += 1
                    last_function_end = j
            
            if last_function_end > 0:
                return last_function_end
        
        # Default: append at end
        return len(lines)
    
    def _add_imports(self, content: str, new_dependencies: List[str]) -> str:
        """Add new import statements at the top of the file"""
        lines = content.split('\n')
        
        # Find where to insert imports (after existing imports or at top)
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                insert_index = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        # Add new imports (avoid duplicates)
        for dep in new_dependencies:
            if dep.strip() and dep not in content:  # Avoid duplicates
                lines.insert(insert_index, dep)
                insert_index += 1
                logger.info(f"[CodeGeneratorAgent] Added import: {dep}")
        
        return '\n'.join(lines)
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response, removing markdown formatting"""
        
        # Remove markdown code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        
        # If no code blocks found, try to extract code-like content
        if not response.strip().startswith(('import ', 'from ', 'def ', 'class ', '#')):
            lines = response.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                stripped = line.strip()
                
                # Start collecting when we see Python-like syntax
                if not in_code and (
                    stripped.startswith(('import ', 'from ', 'def ', 'class ', 'if __name__', '#')) or
                    stripped == '' or 
                    line.startswith('    ') or  # indented line
                    stripped.startswith(('@', 'async def', 'try:', 'with ', 'for ', 'while ', 'if '))
                ):
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
            
            if code_lines:
                response = '\n'.join(code_lines).strip()
        
        return response