import json
import re
import logging
import yaml
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from utils.llm_client import llm_client
from config.agents_io import CodeGeneratorInput, CodeGeneratorOutput, GeneratedFile
import re, os
import boto3
import psycopg2
from botocore.config import Config
import csv
from typing import List, Dict, Any, Set
import difflib

from dotenv import load_dotenv
load_dotenv()

SCHEMA_NAME = "agentic_poc"

logger = logging.getLogger(__name__)

import_string = 'import '
from_string = 'from '
class_string = 'class '






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

    def code_diff_with_comments(self, old_code: str, new_code: str, comment_prefix="# "):
        """
        Compare old code and new code nad return a diff-like output.
        - Added lines start with '+ '
        - Removed lines start with '- ' and are commented
        - Unchanged lines start with '  '

        Parameters:
            old_code (str): The original code as a string.
            new_code (str): The new generated code as a string.
            comment_prefix (str): What to use to comment out removed code.

        Returns:
            str: A combined diff representation.
        """
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()

        diff = difflib.ndiff(old_lines, new_lines)

        result_lines = []
        for line in diff:
            if line.startswith("+ "):
                # Added line
                result_lines.append(f"+ {line[2:]}")
            elif line.startswith("  "):
                # Unchanged
                result_lines.append(f"  {line[2:]}")
            # We ignore "? " lines from ndiff (these are for showing details on changed chars)

        return "\n".join(result_lines)

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
        print("Modification type -----------------", file_info.get("modification_type",""))
        modification_type = file_info.get("modification_type","")
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

        if modification_type == "Migration":
            modified_content = await self.generate_migrated_code_with_rag(current_content, modifications)

        else:
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

        modified_content = self.code_diff_with_comments(current_content, modified_content)

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

            # Separate modifications by type for better processing order
            line_based_mods = []
            other_mods = []

            for mod in modifications:
                if mod.get('line_number', 0) > 0:
                    line_based_mods.append(mod)
                else:
                    other_mods.append(mod)

            # Sort line-based modifications by line number (descending) to avoid offset issues
            line_based_mods.sort(key=lambda x: x.get('line_number', 0), reverse=True)

            # Process line-based modifications first, then others
            all_modifications = line_based_mods + other_mods

            # Track successful modifications for logging
            successful_mods = 0

            for mod in all_modifications:
                action = mod.get('action', '')
                old_code = mod.get('old_code', '').strip()
                new_code = mod.get('new_code', '').strip()
                target_name = mod.get('target_name', '')

                if action == 'add':
                    modified_content = self._add_code_segment(
                        modified_content, new_code, mod.get('line_number', -1), target_name
                    )
                    successful_mods += 1
                    logger.info(f"[CodeGeneratorAgent] Added code segment '{target_name}' in {file_path}")

                elif action == 'modify' and old_code and new_code:
                    if old_code in modified_content:
                        modified_content = modified_content.replace(old_code, new_code, 1)  # Replace only first occurrence
                        successful_mods += 1
                        logger.info(f"[CodeGeneratorAgent] Modified code segment '{target_name}' in {file_path}")
                    else:
                        logger.warning(f"[CodeGeneratorAgent] Old code not found for modification '{target_name}' in {file_path}")
                        return None  # Fall back to LLM

                elif action == 'delete' and old_code:
                    if old_code in modified_content:
                        modified_content = modified_content.replace(old_code, '', 1)  # Remove only first occurrence
                        successful_mods += 1
                        logger.info(f"[CodeGeneratorAgent] Deleted code segment '{target_name}' in {file_path}")
                    else:
                        logger.warning(f"[CodeGeneratorAgent] Old code not found for deletion '{target_name}' in {file_path}")
                        return None  # Fall back to LLM

            # Add new dependencies at the top if any exist
            if new_dependencies:
                modified_content = self._add_imports(modified_content, new_dependencies)

            logger.info(f"[CodeGeneratorAgent] Applied {successful_mods}/{len(modifications)} modifications to {file_path}")
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
            response = self.invoke_claude_sonnet(prompt)

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
                insert_index = self._find_best_insertion_point(lines, new_code)
                lines.insert(insert_index, new_code)
            else:
                # Append at end
                lines.append(new_code)

        return '\n'.join(lines)

    def _find_best_insertion_point(self, lines: List[str], new_code: str) -> int:
        """Find the best place to insert new code based on context"""

        new_code_stripped = new_code.strip()

        # Early return for imports - find first non-import/non-comment line
        if new_code_stripped.startswith((import_string, from_string)):
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith((import_string, from_string, '#')):
                    return i
            return 0

        # For classes - find end of last class
        if new_code_stripped.startswith(class_string):
            return self._find_last_definition_end(lines, class_string)

        # For functions - find end of last function
        if new_code_stripped.startswith(('def ', 'async def ')):
            return self._find_last_definition_end(lines, ('def ', 'async def '))

        # Default: append at end
        return len(lines)

    def _find_last_definition_end(self, lines: List[str], prefixes) -> int:
        """Helper method to find the end of the last class or function definition"""
        if isinstance(prefixes, str):
            prefixes = (prefixes,)

        last_def_end = -1
        i = 0

        while i < len(lines):
            line_stripped = lines[i].strip()

            # Found a definition
            if any(line_stripped.startswith(prefix) for prefix in prefixes):
                i += 1
                # Skip the definition body (indented lines and empty lines)
                while i < len(lines):
                    current_line = lines[i]
                    if current_line.startswith('    ') or current_line.strip() == '':
                        i += 1
                    else:
                        break
                last_def_end = i
            else:
                i += 1

        return last_def_end if last_def_end > 0 else len(lines)

    def _add_imports(self, content: str, new_dependencies: List[str]) -> str:
        """Add new import statements at the top of the file"""
        lines = content.split('\n')
        # Find where to insert imports (after existing imports or at top)
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith((import_string, from_string)):
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

    async def generate_migrated_code_with_rag(self,
                                              original_code: str,
                                              modification_plan: Dict) -> str:

        print("I am inside the migration generator")

        """
        Generate Pyspark 3.5-migrated code using original code, a modification plan, and RAG-provided contexts.
        """
        connection_params = {
            "host" : os.getenv("PG_DB_HOST"),
            "port" : os.getenv("PG_DB_PORT"),
            "dbname" : os.getenv("PG_DB_NAME"),
            "user" : os.getenv("PG_DB_USER"),
            "password" : os.getenv("PG_DB_PASSWORD")
        }
        context = self.analyze_pyspark_code(original_code, connection_params, table_name = "pyspark_data_table")
        context = str(context)
        # context = self.analyze_pyspark_code(original_code)
        # print("printing the context of analyze pyspark")
        print(context)
        print("#"*100)

        combined_context = context

        # Pyspark 3.5 specific migration prompt
        llm_prompt = f"""You are an expert Python developer with deep knowledge of Apache Spark.

            Your task is to migrate the given Pyspark code to be fully compatible with **Pyspark version 3.5**, based on the provided migration plan and relevant technical context.

            Ensure you:
            - Replace deprecated APIs with their new equivalents in 3.5.
            - Follow best practices introduced in Pyspark 3.5 (e.g., new DataFrame operations, SparkSession usage, configuration changes).
            - Preserve all core business logic and comments.
            - Do **not** introduce unnecessary changes or explanations.
            - Output only the fully migrated code in valid Python syntax.
            - Include the commented line at the top of the new generated portion of
              code so the user know where the changes takes place.

            --- Original Code ---
            {original_code}

            --- Migration Plan ---
            {modification_plan}

            --- RAG Context (Pyspark 3.5 Docs / Changes) ---
            {combined_context}

            Return only the updated Python code with all necessary modifications for Pyspark 3.5.
        """

        # Call the LLM to get the migrated code (e.g., OpenAI, vLLM, local model)

        response = self.invoke_claude_sonnet(llm_prompt)
        if response:
            # Clean the response to extract just the code
            cleaned_code = self._extract_code_from_response(response)

        migrated_code = cleaned_code
        return migrated_code

    def extract_code_like_content_line_by_line(self, response):
        lines = response.split('\n')
        code_lines = []
        code_started = False

        for line in lines:
            line_stripped = line.strip()

            # Check if this line indicates start of code
            if not code_started:
                code_indicators = (
                    import_string, from_string, 'def ', class_string, 'if __name__', '#',
                    '@', 'async def', 'try:', 'with ', 'for ', 'while ', 'if '
                )

                if (line_stripped.startswith(code_indicators) or
                    line_stripped == '' or
                    line.startswith('   ')): # indented line
                    code_started = True

            if code_started:
                code_lines.append(line)
        return code_lines

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response, removing markdown formatting"""

        # Handle markdown code blocks first
        code_block_patterns = [
            ("```python", "```"),
            ("```", "```")
        ]

        for start_marker, end_marker in code_block_patterns:
            if start_marker in response:
                start_idx = response.find(start_marker) + len(start_marker)
                end_idx = response.find(end_marker, start_idx)
                if end_idx != -1:
                    return response[start_idx:end_idx].strip()

        # If no markdown blocks, check if response already looks like Python code
        response_stripped = response.strip()
        python_indicators = (import_string, from_string, 'def ', class_string, '#', 'if __name__')

        if response_stripped.startswith(python_indicators):
            return response_stripped

        code_lines = self.extract_code_like_content_line_by_line(response)

        return '\n'.join(code_lines).strip() if code_lines else response_stripped

    def clean_sphinx(self, text: str) -> str:
        """Remove Sphinx/ReST markup like :meth:`...` or :class:`...`."""
        if not text:
            return ""
        cleaned = re.sub(r':\w+:`([^`]+)`', r'\1', text)
        cleaned = cleaned.replace("``","")
        return cleaned.strip()


    def load_pyspark_details_from_db(
            self, conn_params: Dict[str, Any],
            table_name: str = "pyspark_data_table"
    ) -> (Dict[str, Dict[str, str]], Set[str], Set[str]):
        print("Connection to DB and fetching Pyspark API details...")
        details = {}
        modules = set()
        class_names = set()

        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        cursor.execute(f"SELECT * from {SCHEMA_NAME}.{table_name};")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        for row in rows:
            row_dict = dict(zip(columns, row))
            qualname = row_dict["qualname"].strip()
            details[qualname] = row_dict

            modules.add(row_dict["module"].strip())

            parts = qualname.split(".")
            if len(parts) > 2:
                class_names.add(parts[-2])

        cursor.close()
        conn.close()
        return details, modules, class_names


    def has_pyspark_import(self, code_content: str) -> bool:
        """Check if code imports or references pyspark"""
        return bool(re.search(r'import\s+pyspark|from\s+pyspark', code_content))


    def extract_pyspark_calls(self, code_content: str) -> List[str]:
        """Extract possible Pyspark calls from code using regex."""
        return re.findall(r'\b(?:pyspark(?:\.\w+)+|\w+\.\w+)\b', code_content)


    def generate_full_paths(self, call: str, known_modules: Set[str], known_classes: Set[str]) -> List[str]:
        """Given a call, generate full possible qualified names to check in details"""
        full_paths = []
        if call.startswith("pyspark."):
            full_paths.append(call)
        else:
            method_name = call.split('.')[-1]
            full_paths.extend(f"{mod}.{method_name}" for mod in known_modules)
            full_paths.extend(f"{mod}.{cls}.{method_name}" for mod in known_modules for cls in known_classes)
        return full_paths


    def is_significant_change(self, details: dict) -> bool:
        def norm(val):
            return str(val or "").strip().lower()

        deprecated_flag = norm(details.get('deprecated')) == 'true'
        deprecated_note = norm(details.get('deprecated_note'))
        versionchanged_note = norm(details.get('versionchanged_note'))
        doc = norm(details.get('docstring'))

        deprecated_keywords = [
            'deprecated', 'use ', 'replaced by',
            'will be removed', 'alias for', 'moved to'
        ]

        found_in_notes = (
            any(kw in deprecated_note for kw in deprecated_keywords) or
            any(kw in versionchanged_note for kw in deprecated_keywords) or
            any(kw in doc for kw in deprecated_keywords)
        )

        return deprecated_flag or found_in_notes


    def fetch_pyspark_calls(
            self, pyspark_calls: List[str],
            pyspark_details: Dict[str, Dict[str, str]],
            known_modules: Set[str],
            known_classes: Set[str],
    ) ->List[Dict[str, Any]]:
        changes = []
        checked_functions = set()

        for call in pyspark_calls:
            full_paths = self.generate_full_paths(call, known_modules, known_classes)

            for full_path in full_paths:
                if full_path in checked_functions:
                    continue
                checked_functions.add(full_path)

                if full_path in pyspark_details:
                    details = pyspark_details[full_path]
                    if self.is_significant_change(details):
                        change_entry = {
                            "function_name" : full_path,
                            "reason" : "",
                            "deprecated" : details.get("deprecated"),
                            "parameter" : details.get("parameters", "passing parameters"),
                            "summary" : details.get("docstring")
                        }
                        if details.get("deprecated") == "True":
                            change_entry["reason"] = f"Deprecated in Pyspark {details.get('deprecated_in', '3.5')}"
                            change_entry["suggestion"] = details.get("deprecated_note")
                        elif details.get("versionchanged"):
                            change_entry["reason"] = f"Changed in Pyspark {details['versionchanged']}"
                            change_entry["suggestion"] = self.clean_sphinx(details.get("versionchanged_note", "Check the documentation for updates."))
                        changes.append(change_entry)
                    break

        return changes


    def analyze_pyspark_code(self, code_content:str, conn_params: Dict[str, Any], table_name: str = "pyspark_data_table") -> Dict[str, Any]:
        pyspark_details, known_modules, known_classes = self.load_pyspark_details_from_db(conn_params, table_name)

        if not self.has_pyspark_import(code_content):
            return {
                "has_pyspark": False,
                "aligned_with_3_5": True,
                "needs_modification": False,
                "changes": []
            }

        pyspark_calls = self.extract_pyspark_calls(code_content)
        print("PYSPARK FOUND:", pyspark_calls)

        changes = self.fetch_pyspark_calls(pyspark_calls, pyspark_details, known_modules, known_classes)

        return {
            "has_pyspark": True,
            "aligned_with_3_5": len(changes) == 0,
            "needs_modification": len(changes) > 0,
            "changes": changes
        }

    def invoke_claude_sonnet(self, prompt, max_tokens=10000):
        # Model ID for Claude 3.7 Sonnet
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        service_name = os.getenv("AWS_SERVICE_NAME","")
        region_name = os.getenv("AWS_REGION_NAME","")
        config = Config(read_timeout=10000)
        bedrock_runtime = boto3.client(
            service_name=service_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=config
        )
        model_id = os.getenv("AWS_MODEL_ID", "")

        # Prepare the request body
        body = {
            "anthropic_version" : "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            # Invoke the model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json'
            )

            # Parse the response
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']

        except Exception as e:
            print(f"Error: {e}")
            return None