"""
Simple Document Generator - Clean Multi-Level Vector Search

Enhanced with LLM intelligence:
1. Scan repository (files/classes/functions)
2. LLM enhance each block with docstrings/comments
3. Create single semantic embedding per block
4. Smart search: LLM query analysis + metadata + vector similarity
5. LLM result refinement and ranking
6. Return optimized results

Clean, fast, intelligent.
"""

import asyncio
import ast
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, cast
from pathlib import Path
import time
import json
import requests
import openai
from simple_database_schema import  create_simple_database


@dataclass
class QueryAnalysis:
    """LLM analysis of user query."""
    intent: str  # "create", "modify", "fix", "understand", etc.
    key_terms: List[str]  # Important technical terms
    context: str  # What user is trying to accomplish
    search_focus: str  # "functions", "classes", "files", "all"
    confidence: float  # How clear the query is


@dataclass
class SearchResult:
    """Simple search result."""
    file_path: str
    name: str  # function/class/file name
    type: str  # "file", "class", "function"
    relevance_score: float
    enhanced_content: str = ""
    parent_class: Optional[str] = None
    match_reason: str = ""  # Why this result is relevant
    rich_metadata: Optional[str] = None  # Added for rich metadata storage


@dataclass
class CodeBlock:
    """Code block for processing."""
    content: str
    file_path: str
    type: str  # "file", "class", "function"
    name: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    parent_class: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SimpleDocumentGenerator:
    """
    Simple Document Generator - Your Original Design

    Clean, fast, effective:
    1. Multi-level indexing (file/class/function)
    2. LLM enhancement for better embeddings
    3. Hybrid search (metadata + vector)
    4. Fast results without complexity
    """

    def __init__(self,
                 database_config: Dict[str, Any]):
        """
        Initialize simple document generator.

        Args:
            database_config: PostgreSQL connection config
            embedding_client: OpenAI embedding client
            llm_client: LLM client for enhancement
        """
        self.database = create_simple_database(database_config)
        self.logger = logging.getLogger(__name__)

        # Simple config
        self.supported_extensions = {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c'}
        self.batch_size = 5

    async def index_repository(self, repo_path: str, repo_id: str) -> Dict[str, Any]:
        """
        Index repository with your simple approach.

        1. Scan repository structure
        2. Extract code blocks (files/classes/functions)
        3. LLM enhance each block
        4. Create embeddings
        5. Store in database
        """
        start_time = time.time()
        self.logger.info(f"🚀 Starting simple indexing for: {repo_path}")
        print(f"🚀 Starting simple indexing for: {repo_path}")

        try:
            # Step 1: Scan repository
            print(f"📁 Step 1: Scanning repository structure...")
            files = await self._scan_repository(repo_path)
            self.logger.info(f"📁 Found {len(files)} files to process")
            print(f"📁 Found {len(files)} files to process")

            if not files:
                return {"success": False, "error": "No supported files found"}

            # Step 2: Extract code blocks
            print(f"🔍 Step 2: Extracting code blocks from {len(files)} files...")
            all_blocks = []
            files_processed = 0

            for i, file_path in enumerate(files, 1):
                try:
                    print(f"📄 Processing file {i}/{len(files)}: {file_path}")
                    blocks = await self._extract_code_blocks(file_path)
                    print("index repo",file_path)
                    all_blocks.extend(blocks)
                    files_processed += 1

                    # Show progress every 5 files
                    if i % 5 == 0:
                        print(f"📊 Progress: {i}/{len(files)} files processed ({len(all_blocks)} blocks so far)")

                except Exception as e:
                    self.logger.warning(f"Failed to process file {file_path}: {e}")
                    print(f"⚠️ Warning: Failed to process {file_path}: {e}")
                    continue

            self.logger.info(f"🔍 Extracted {len(all_blocks)} code blocks from {files_processed} files")
            print(f"🔍 Extracted {len(all_blocks)} code blocks from {files_processed} files")

            # Step 3: Process in batches
            print(f"🤖 Step 3: Processing {len(all_blocks)} blocks with LLM enhancement...")
            processed_count = 0
            failed_count = 0

            for i in range(0, len(all_blocks), self.batch_size):
                batch = all_blocks[i:i + self.batch_size]
                batch_start = time.time()

                print(f"🔄 Processing batch {i//self.batch_size + 1}/{(len(all_blocks) + self.batch_size - 1)//self.batch_size} ({len(batch)} blocks)")

                try:
                    await self._process_batch(batch, repo_id, repo_path)
                    processed_count += len(batch)
                    batch_time = time.time() - batch_start
                    print(f"✅ Batch completed in {batch_time:.2f}s ({processed_count}/{len(all_blocks)} total)")

                except Exception as batch_error:
                    failed_count += len(batch)
                    self.logger.error(f"Batch processing failed: {batch_error}")
                    print(f"❌ Batch failed: {batch_error}")
                    continue

                # Progress update every few batches
                if (i // self.batch_size) % 4 == 0:
                    elapsed = time.time() - start_time
                    avg_time_per_block = elapsed / max(processed_count, 1)
                    remaining_blocks = len(all_blocks) - processed_count - failed_count
                    estimated_remaining = remaining_blocks * avg_time_per_block

                    print(f"📊 Progress: {processed_count}/{len(all_blocks)} blocks processed")
                    print(f"⏱️ Elapsed: {elapsed:.1f}s, Estimated remaining: {estimated_remaining:.1f}s")

                # Small delay to respect rate limits
                await asyncio.sleep(0.1)

            processing_time = time.time() - start_time

            success_rate = (processed_count / len(all_blocks)) * 100 if all_blocks else 0

            self.logger.info(f"🎯 Simple indexing completed: {files_processed} files, {processed_count} blocks in {processing_time:.2f}s")
            print(f"🎯 Simple indexing completed!")
            print(f"📊 Files processed: {files_processed}/{len(files)}")
            print(f"📊 Blocks processed: {processed_count}/{len(all_blocks)} ({success_rate:.1f}% success)")
            print(f"⏱️ Total time: {processing_time:.2f} seconds")

            return {
                "success": True,
                "files_processed": files_processed,
                "total_blocks": processed_count,
                "blocks_processed": processed_count,  # For compatibility
                "failed_blocks": failed_count,
                "processing_time": processing_time,
                "success_rate": success_rate
            }

        except Exception as e:
            error_msg = f"Repository indexing failed: {e}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "files_processed": 0,
                "total_blocks": 0,
                "processing_time": time.time() - start_time
            }


    async def _scan_repository(self, repo_path: str) -> List[str]:
        """Scan repository for supported files."""
        files = []

        for root, dirs, filenames in os.walk(repo_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv'}]

            for filename in filenames:
                if any(filename.endswith(ext) for ext in self.supported_extensions):
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
                    print(f"file_path: {file_path}")

        return files

    async def _extract_code_blocks(self, file_path: str) -> List[CodeBlock]:
        """Extract file/class/function blocks from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            blocks = []

            # File-level block
            file_block = CodeBlock(
                content=content,
                file_path=file_path,
                type="file",
                name=os.path.basename(file_path),
                metadata=self._extract_file_metadata(content, file_path)
            )
            blocks.append(file_block)

            # Extract classes and functions for Python files
            if file_path.endswith('.py'):
                blocks.extend(self._extract_python_blocks(content, file_path))

            return blocks

        except Exception as e:
            self.logger.warning(f"Failed to extract blocks from {file_path}: {e}")
            return []

    def _extract_file_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from file content including rich metadata."""
        lines = content.split('\n')

        # Extract basic metadata (existing logic)
        imports = []
        function_names = []
        class_names = []

        for line in lines:
            line = line.strip()

            # Python imports
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)

            # Python functions
            func_match = re.match(r'def\s+(\w+)', line)
            if func_match:
                function_names.append(func_match.group(1))

            # Python classes
            class_match = re.match(r'class\s+(\w+)', line)
            if class_match:
                class_names.append(class_match.group(1))

        # 🆕 ENHANCED: Add file-level rich metadata
        file_rich_metadata = self._extract_file_level_rich_metadata(content, file_path)

        # Combine basic metadata with rich metadata
        basic_metadata = {
            'language': self._detect_language(file_path),
            'file_size': len(content),
            'line_count': len(lines),
            'imports': imports[:10],  # Limit to avoid huge metadata
            'function_names': function_names,
            'class_names': class_names,
            'file_hash': hashlib.md5(content.encode()).hexdigest()
        }

        # 🆕 Merge basic and rich metadata
        return {**basic_metadata, **file_rich_metadata}

    def _extract_file_level_rich_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract rich metadata for file-level analysis."""
        try:
            # File-level intent classification based on entire file content
            file_intent = self._classify_file_intent(content, file_path)

            # File-level quality assessment
            file_quality = self._assess_file_quality(content, file_path)

            # File architecture and patterns
            file_patterns = self._analyze_file_patterns(content, file_path)

            # File complexity assessment
            file_complexity = self._assess_file_complexity(content, file_path)

            return {
                **file_intent,
                **file_quality,
                **file_patterns,
                "complexity_level": file_complexity,
                "file_type": self._classify_file_type(file_path),
                "architectural_layer": self._classify_architectural_layer(file_path),
                "maintainability_score": self._calculate_file_maintainability(content, file_path)
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract file-level rich metadata: {e}")
            return {}

    def _classify_file_intent(self, content: str, file_path: str) -> Dict[str, Any]:
        """Classify the main purpose and intent of the entire file."""
        # Enhanced intent patterns for file-level analysis
        intent_patterns = {
            "authentication": ["login", "auth", "verify", "validate", "credential", "token", "session", "user", "password"],
            "data_processing": ["process", "transform", "parse", "convert", "format", "serialize", "data", "etl"],
            "api_endpoint": ["route", "endpoint", "handler", "controller", "request", "response", "api", "rest"],
            "database": ["query", "insert", "update", "delete", "model", "schema", "db", "sql", "orm"],
            "validation": ["validate", "check", "verify", "ensure", "assert", "sanitize", "rules"],
            "error_handling": ["error", "exception", "handle", "catch", "raise", "fail", "logging"],
            "utility": ["helper", "util", "tool", "format", "convert", "common", "shared"],
            "test": ["test_", "_test", "assert", "mock", "fixture", "spec", "unittest"],
            "configuration": ["config", "setting", "env", "constant", "default", "settings"],
            "security": ["secure", "encrypt", "decrypt", "hash", "permission", "access", "auth"],
            "user_management": ["user", "account", "profile", "registration", "signup", "management"],
            "payment": ["payment", "billing", "charge", "invoice", "transaction", "stripe", "paypal"],
            "notification": ["notify", "email", "sms", "alert", "message", "send", "communication"],
            "file_operations": ["file", "upload", "download", "read", "write", "storage", "filesystem"],
            "logging": ["log", "logger", "debug", "info", "warn", "error", "audit", "monitoring"]
        }

        # Analyze content and file path
        content_lower = content.lower()
        file_path_lower = file_path.lower()

        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = 0
            for keyword in keywords:
                # File path matches (highest weight for files)
                if keyword in file_path_lower:
                    score += 5
                # Content frequency (weighted by occurrence)
                content_count = content_lower.count(keyword)
                score += min(content_count, 10)  # Cap at 10 to avoid skewing
            intent_scores[intent] = score

        # Get primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])

        # Extract secondary intents (for multi-purpose files)
        secondary_intents = [intent for intent, score in intent_scores.items()
                           if score > 5 and intent != primary_intent[0]][:3]

        # Classify operation pattern
        operation_pattern = self._classify_file_operation_pattern(content)

        return {
            "intent_category": primary_intent[0] if primary_intent[1] > 0 else "general",
            "intent_confidence": min(primary_intent[1] / 20.0, 1.0),  # Normalize to 0-1
            "secondary_intents": secondary_intents,
            "operation_pattern": operation_pattern,
            "domain_area": self._classify_domain_area(file_path)
        }

    def _classify_file_operation_pattern(self, content: str) -> str:
        """Classify the overall operation pattern of the file."""
        content_lower = content.lower()

        # Count different operation types
        create_ops = content_lower.count("create") + content_lower.count("add") + content_lower.count("insert")
        read_ops = content_lower.count("get") + content_lower.count("find") + content_lower.count("retrieve")
        update_ops = content_lower.count("update") + content_lower.count("modify") + content_lower.count("edit")
        delete_ops = content_lower.count("delete") + content_lower.count("remove") + content_lower.count("destroy")

        # Determine primary pattern
        operations = {
            "crud": create_ops + read_ops + update_ops + delete_ops,
            "data_processing": content_lower.count("process") + content_lower.count("transform"),
            "service_layer": content_lower.count("service") + content_lower.count("business"),
            "utility": content_lower.count("util") + content_lower.count("helper"),
            "configuration": content_lower.count("config") + content_lower.count("setting")
        }

        primary_pattern = max(operations.items(), key=lambda x: x[1])
        return primary_pattern[0] if primary_pattern[1] > 0 else "general"

    def _assess_file_quality(self, content: str, file_path: str) -> Dict[str, Any]:
        """Assess overall file quality indicators."""
        try:
            # Count various quality indicators
            has_error_handling = self._check_error_handling(content)
            has_logging = self._check_logging(content)
            has_docstrings = content.count('"""') > 2 or content.count("'''") > 2
            has_type_hints = ":" in content and any(hint in content for hint in ["str", "int", "bool", "List", "Dict"])

            # Assess documentation coverage
            total_functions = content.count("def ")
            documented_functions = content.count('"""') + content.count("'''")
            doc_coverage = documented_functions / max(total_functions, 1)

            # Check for imports organization
            import_section = "\n".join(content.split("\n")[:50])  # First 50 lines
            has_organized_imports = "import" in import_section and not any(
                line.strip().startswith("import") for line in content.split("\n")[100:]  # No imports after line 100
            )

            return {
                "has_error_handling": has_error_handling,
                "has_logging": has_logging,
                "has_docstrings": has_docstrings,
                "has_type_hints": has_type_hints,
                "documentation_coverage": min(doc_coverage, 1.0),
                "has_organized_imports": has_organized_imports,
                "documentation_quality": self._assess_file_documentation_quality(content)
            }
        except Exception as e:
            self.logger.warning(f"Failed to assess file quality: {e}")
            return {}

    def _analyze_file_patterns(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze architectural and design patterns in the file."""
        try:
            # Detect design patterns
            design_patterns = []
            if "class" in content and "def __init__" in content:
                design_patterns.append("object_oriented")
            if "def " in content and "class" not in content:
                design_patterns.append("functional")
            if "async def" in content:
                design_patterns.append("async_programming")
            if "yield" in content:
                design_patterns.append("generator_pattern")

            # Security patterns
            security_patterns = self._detect_security_patterns(content)

            # Performance patterns
            performance_indicators = self._detect_performance_patterns(content)

            # Framework patterns
            framework_patterns = []
            frameworks = {
                "django": ["django", "models.Model", "views.View"],
                "flask": ["flask", "app.route", "@app."],
                "fastapi": ["fastapi", "APIRouter", "Depends"],
                "pytest": ["pytest", "test_", "@pytest."],
                "sqlalchemy": ["sqlalchemy", "Column", "relationship"]
            }

            content_lower = content.lower()
            for framework, indicators in frameworks.items():
                if any(indicator.lower() in content_lower for indicator in indicators):
                    framework_patterns.append(framework)

            return {
                "design_patterns": design_patterns,
                "security_patterns": security_patterns,
                "performance_indicators": performance_indicators,
                "framework_patterns": framework_patterns,
                "code_smells": self._detect_file_code_smells(content)
            }
        except Exception as e:
            self.logger.warning(f"Failed to analyze file patterns: {e}")
            return {}

    def _assess_file_complexity(self, content: str, file_path: str) -> str:
        """Assess overall file complexity level."""
        try:
            lines = content.split('\n')
            line_count = len(lines)

            # Count structural elements
            function_count = content.count("def ")
            class_count = content.count("class ")
            import_count = content.count("import ") + content.count("from ")

            # Calculate complexity score
            complexity_score = 0

            # Size factor
            if line_count > 500:
                complexity_score += 3
            elif line_count > 200:
                complexity_score += 2
            elif line_count > 100:
                complexity_score += 1

            # Structure factor
            if function_count > 20:
                complexity_score += 2
            elif function_count > 10:
                complexity_score += 1

            if class_count > 5:
                complexity_score += 2
            elif class_count > 2:
                complexity_score += 1

            # Dependency factor
            if import_count > 20:
                complexity_score += 2
            elif import_count > 10:
                complexity_score += 1

            # Control flow complexity
            control_structures = (content.count("if ") + content.count("for ") +
                                content.count("while ") + content.count("try:"))
            if control_structures > 50:
                complexity_score += 2
            elif control_structures > 20:
                complexity_score += 1

            # Map score to level
            if complexity_score <= 2:
                return "low"
            elif complexity_score <= 5:
                return "medium"
            elif complexity_score <= 8:
                return "high"
            else:
                return "very_high"

        except Exception as e:
            self.logger.warning(f"Failed to assess file complexity: {e}")
            return "medium"

    def _classify_file_type(self, file_path: str) -> str:
        """Classify the type of file based on naming conventions."""
        filename = os.path.basename(file_path).lower()

        if filename.startswith("test_") or filename.endswith("_test.py") or "test" in filename:
            return "test"
        elif filename in ["__init__.py", "main.py", "app.py", "run.py"]:
            return "entry_point"
        elif "model" in filename:
            return "model"
        elif "view" in filename or "template" in filename:
            return "view"
        elif "controller" in filename or "handler" in filename:
            return "controller"
        elif "service" in filename:
            return "service"
        elif "util" in filename or "helper" in filename:
            return "utility"
        elif "config" in filename or "setting" in filename:
            return "configuration"
        else:
            return "module"

    def _classify_architectural_layer(self, file_path: str) -> str:
        """Classify which architectural layer this file belongs to."""
        path_lower = file_path.lower()

        if any(layer in path_lower for layer in ["model", "entity", "data"]):
            return "data_layer"
        elif any(layer in path_lower for layer in ["view", "template", "ui", "frontend"]):
            return "presentation_layer"
        elif any(layer in path_lower for layer in ["controller", "handler", "api", "endpoint"]):
            return "controller_layer"
        elif any(layer in path_lower for layer in ["service", "business", "logic"]):
            return "business_layer"
        elif any(layer in path_lower for layer in ["util", "helper", "common", "shared"]):
            return "utility_layer"
        elif any(layer in path_lower for layer in ["config", "setting"]):
            return "configuration_layer"
        else:
            return "application_layer"

    def _assess_file_documentation_quality(self, content: str) -> str:
        """Assess the overall documentation quality of the file."""
        total_lines = len(content.split('\n'))
        comment_lines = len([line for line in content.split('\n') if line.strip().startswith('#')])
        docstring_count = content.count('"""') + content.count("'''")

        if docstring_count == 0 and comment_lines < total_lines * 0.05:
            return "none"
        elif docstring_count < 2 and comment_lines < total_lines * 0.1:
            return "minimal"
        elif docstring_count >= 2 or comment_lines >= total_lines * 0.1:
            return "good"
        else:
            return "basic"

    def _detect_file_code_smells(self, content: str) -> List[str]:
        """Detect code smells at file level."""
        smells = []
        lines = content.split('\n')

        # Large file smell
        if len(lines) > 500:
            smells.append("large_file")

        # Too many functions
        function_count = content.count("def ")
        if function_count > 30:
            smells.append("too_many_functions")

        # Too many classes
        class_count = content.count("class ")
        if class_count > 10:
            smells.append("too_many_classes")

        # Duplicate code patterns (simplified)
        if content.count("if __name__ == '__main__'") > 1:
            smells.append("duplicate_main_blocks")

        # Poor separation of concerns
        if "database" in content.lower() and "html" in content.lower():
            smells.append("mixed_concerns")

        return smells[:3]  # Top 3 most relevant

    def _calculate_file_maintainability(self, content: str, file_path: str) -> float:
        """Calculate file-level maintainability score."""
        try:
            score = 1.0

            # Size penalty
            line_count = len(content.split('\n'))
            if line_count > 500:
                score -= 0.3
            elif line_count > 300:
                score -= 0.2
            elif line_count > 200:
                score -= 0.1

            # Documentation bonus
            has_docstrings = content.count('"""') > 0 or content.count("'''") > 0
            if has_docstrings:
                score += 0.1

            # Error handling bonus
            if self._check_error_handling(content):
                score += 0.1

            # Organization bonus
            import_section = "\n".join(content.split("\n")[:20])
            if "import" in import_section:
                score += 0.05

            # Complexity penalty
            complexity = self._assess_file_complexity(content, file_path)
            complexity_penalty = {
                "low": 0.0,
                "medium": 0.1,
                "high": 0.2,
                "very_high": 0.3
            }
            score -= complexity_penalty.get(complexity, 0.1)

            return max(0.0, min(1.0, score))
        except Exception as e:
            self.logger.warning(f"Failed to calculate file maintainability: {e}")
            return 0.5

    def _extract_python_blocks(self, content: str, file_path: str) -> List[CodeBlock]:
        """Extract Python classes and functions with rich metadata."""
        blocks = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_content = ast.get_source_segment(content, node)
                    if class_content:
                        # 🆕 ENHANCED CLASS METADATA
                        class_metadata = {
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                            'base_classes': [getattr(base, 'id', getattr(base, 'attr', str(base))) for base in node.bases],
                            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                            'docstring': ast.get_docstring(node) or "",
                            'lines_of_code': (node.end_lineno - node.lineno) if node.end_lineno else 1,
                            **self._classify_code_intent(class_content, file_path, node.name)
                        }

                        blocks.append(CodeBlock(
                            content=class_content,
                            file_path=file_path,
                            type="class",
                            name=node.name,
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            metadata=class_metadata
                        ))

                elif isinstance(node, ast.FunctionDef):
                    func_content = ast.get_source_segment(content, node)
                    if func_content:
                        parent_class = None
                        # Simple check if function is inside a class
                        for parent in ast.walk(tree):
                            if isinstance(parent, ast.ClassDef):
                                if (parent.lineno <= node.lineno <= (parent.end_lineno or float('inf'))):
                                    parent_class = parent.name
                                    break

                        # 🆕 ENHANCED FUNCTION METADATA
                        function_metadata = {
                            **self._extract_signature_metadata(node, func_content),
                            **self._classify_code_intent(func_content, file_path, node.name),
                            **self._extract_quality_metadata(func_content, file_path, node)
                        }

                        blocks.append(CodeBlock(
                            content=func_content,
                            file_path=file_path,
                            type="function",
                            name=node.name,
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            parent_class=parent_class,
                            metadata=function_metadata
                        ))

        except Exception as e:
            self.logger.warning(f"Failed to parse Python AST for {file_path}: {e}")

        return blocks

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c'
        }
        return lang_map.get(ext, 'unknown')

    async def _process_batch(self, blocks: List[CodeBlock], repo_id: str, repo_path: str):
        """Process a batch of code blocks."""
        for block in blocks:
            try:
                # Step 1: LLM enhance the code
                enhanced_content = await self._enhance_with_llm(block)

                # Step 2: Create embedding
                embedding = await self._create_embedding(enhanced_content)

                # Step 3: Store in database
                await self._store_block(block, repo_id, enhanced_content, embedding, repo_path)

            except Exception as e:
                self.logger.warning(f"Failed to process block {block.name} in {block.file_path}: {e}")

        # Small delay to respect rate limits
        await asyncio.sleep(0.1)

    async def _enhance_with_llm(self, block: CodeBlock) -> str:
        """Enhance code block with LLM-generated documentation."""
        try:
            prompt = f"""Add helpful docstring/comments to this {block.type}:

            {block.content}

            Add a clear docstring that explains:
            1. What this {block.type} does
            2. Key parameters/attributes (if any)
            3. Return value/purpose (if applicable)
            4. Any important usage notes

            Return the enhanced code with good documentation:
            """

            base_url = os.getenv("TIGER_BASE_URL")
            api_key = os.getenv("TIGER_API_KEY")
            model = os.getenv("TIGER_MODEL_NAME")

            client = openai.OpenAI(api_key =  api_key,
                                base_url = base_url)
            response = client.chat.completions.create(model=model,
                                                    messages = [{"role": "user",
                                                                "content": prompt
                                                                }
                                                                ])
            enhanced = response.choices[0].message.content

            # Fallback if LLM returns something weird
            if len(enhanced) < len(block.content) * 0.8:
                return block.content

            return enhanced

        except Exception as e:
            self.logger.warning(f"LLM enhancement failed for {block.name}: {e}")
            return block.content

    async def _create_embedding(self, content: str) -> List[float]:
        url = os.getenv("TIGER_EMBEDDING_BASE_URL")
        api_key = os.getenv("TIGER_EMBEDDING_API_KEY")
        model = os.getenv("TIGER_EMBEDDING_MODEL_NAME")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {"model": model, "input": content}
        try:
            response = requests.post(url, json=data, headers=headers, verify=False)
            return response.json()['data'][0]['embedding']
        except Exception as e:
            self.logger.warning(f"Embedding creation failed: {e}")
            return [0.0] * 1536

    def _get_relative_path(self, absolute_path: str, repo_path: str) -> str:
        """Convert absolute path to clean relative path from repo root."""
        print("I am inside _get_relative_path")
        print(f"absolute_path is :{absolute_path}")
        print(f"repo_path is :{repo_path}")
        print(f"Output : {os.path.relpath(absolute_path, repo_path)}")
        repo_path = '/'.join(repo_path.split("/")[:-1])
        print(f"repo_path is :{repo_path}")
        try:
            return os.path.relpath(absolute_path, repo_path)
        except:
            print("Inside except of _get_relative_path function")
            # Fallback: if can't get relative path, use basename
            return os.path.basename(absolute_path)

    async def _store_block(self, block: CodeBlock, repo_id: str,
                          enhanced_content: str, embedding: List[float], repo_path):
        """Store code block in appropriate database table."""
        try:
            # Convert absolute path to clean relative path
            # repo_path = getattr(self, '_current_repo_path', os.path.dirname(block.file_path))
            print("Inside _store_block")
            print(f"repo_path : {repo_path}")

            relative_file_path = self._get_relative_path(block.file_path, repo_path)
            print(f"relative_file_path : {relative_file_path}")


            if block.type == "file":
                await self.database.store_file_embedding(
                    repo_id, relative_file_path, embedding,
                    enhanced_content, block.content, block.metadata
                )
            elif block.type == "class":
                # ✅ FIXED: Pass ALL metadata including rich metadata
                class_storage_metadata = {
                    'methods': block.metadata.get('methods', []),
                    'start_line': block.start_line,
                    'end_line': block.end_line,
                    **block.metadata  # ✅ Include ALL rich metadata
                }
                await self.database.store_class_embedding(
                    repo_id, relative_file_path, block.name, embedding,
                    enhanced_content, block.content, class_storage_metadata
                )
            elif block.type == "function":
                # ✅ FIXED: Pass ALL metadata including rich metadata
                function_storage_metadata = {
                    'parent_class': block.parent_class,
                    'parameters': block.metadata.get('parameters', []),
                    'start_line': block.start_line,
                    'end_line': block.end_line,
                    **block.metadata  # ✅ Include ALL rich metadata (signatures, intent, quality)
                }
                await self.database.store_function_embedding(
                    repo_id, relative_file_path, block.name, embedding,
                    enhanced_content, block.content, function_storage_metadata
                )
        except Exception as e:
            self.logger.error(f"Failed to store {block.type} {block.name}: {e}")


    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query for metadata search."""
        # Simple term extraction - split on whitespace and remove common words
        terms = query.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        meaningful_terms = []
        for term in terms:
            # Remove punctuation
            clean_term = re.sub(r'[^\w]', '', term)
            if clean_term and len(clean_term) > 2 and clean_term not in stop_words:
                meaningful_terms.append(clean_term)

        return meaningful_terms


    async def close(self):
        """Close database connections."""
        await self.database.close()

    # 🆕 RICH METADATA EXTRACTION METHODS

    def _extract_signature_metadata(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Extract detailed function signature information."""
        try:
            metadata = {
                "parameters": [arg.arg for arg in node.args.args],
                "parameter_count": len(node.args.args),
                "has_self": len(node.args.args) > 0 and node.args.args[0].arg == "self",
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                "docstring": ast.get_docstring(node) or "",
                "lines_of_code": (node.end_lineno - node.lineno) if node.end_lineno else 1,
                "return_type": self._extract_return_type_hint(node),
                "parameter_types": self._extract_parameter_types(node)
            }

            # Extract function signature string
            signature_parts = []
            if metadata["has_self"]:
                signature_parts.append("self")
            signature_parts.extend([f"{param}" for param in metadata["parameters"][1 if metadata["has_self"] else 0:]])
            metadata["signature"] = f"{node.name}({', '.join(signature_parts)})"

            return metadata
        except Exception as e:
            self.logger.warning(f"Failed to extract signature metadata: {e}")
            return {}

    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name safely."""
        if hasattr(decorator, 'id'):
            return decorator.id
        elif hasattr(decorator, 'attr'):
            return decorator.attr
        else:
            return str(decorator)

    def _get_base_class_name(self, base) -> str:
        """Extract base class name safely from AST node."""
        if hasattr(base, 'id'):
            return base.id  # ast.Name node
        elif hasattr(base, 'attr'):
            return base.attr  # ast.Attribute node
        else:
            try:
                return ast.unparse(base) if hasattr(ast, 'unparse') else str(base)
            except:
                return "unknown"

    def _extract_return_type_hint(self, node: ast.FunctionDef) -> str:
        """Extract return type hint if available."""
        try:
            if node.returns:
                return ast.unparse(node.returns) if hasattr(ast, 'unparse') else "unknown"
        except:
            pass
        return "unknown"

    def _extract_parameter_types(self, node: ast.FunctionDef) -> Dict[str, str]:
        """Extract parameter type hints."""
        param_types = {}
        try:
            for arg in node.args.args:
                if arg.annotation:
                    param_types[arg.arg] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else "unknown"
        except:
            pass
        return param_types

    def _classify_code_intent(self, content: str, file_path: str, function_name: str) -> Dict[str, Any]:
        """Classify the main purpose and intent of this code."""
        # Intent classification patterns
        intent_patterns = {
            "authentication": ["login", "auth", "verify", "validate", "credential", "token", "session"],
            "data_processing": ["process", "transform", "parse", "convert", "format", "serialize"],
            "api_endpoint": ["route", "endpoint", "handler", "controller", "request", "response"],
            "database": ["query", "insert", "update", "delete", "model", "schema", "db", "sql"],
            "validation": ["validate", "check", "verify", "ensure", "assert", "sanitize"],
            "error_handling": ["error", "exception", "handle", "catch", "raise", "fail"],
            "utility": ["helper", "util", "tool", "format", "convert", "common"],
            "test": ["test_", "_test", "assert", "mock", "fixture", "spec"],
            "configuration": ["config", "setting", "env", "constant"],
            "security": ["secure", "encrypt", "decrypt", "hash", "permission", "access"],
            "user_management": ["user", "account", "profile", "registration", "signup"],
            "payment": ["payment", "billing", "charge", "invoice", "transaction"],
            "notification": ["notify", "email", "sms", "alert", "message", "send"],
            "file_operations": ["file", "upload", "download", "read", "write", "storage"],
            "logging": ["log", "logger", "debug", "info", "warn", "error", "audit"]
        }

        # Determine primary intent
        content_lower = content.lower()
        function_name_lower = function_name.lower()
        file_path_lower = file_path.lower()

        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = 0
            for keyword in keywords:
                # Function name matches (highest weight)
                if keyword in function_name_lower:
                    score += 3
                # File path matches (medium weight)
                if keyword in file_path_lower:
                    score += 2
                # Content matches (lower weight)
                if keyword in content_lower:
                    score += 1
            intent_scores[intent] = score

        # Get primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])

        # Extract additional semantic tags
        tags = []
        for intent, score in intent_scores.items():
            if score > 0:
                tags.append(intent)

        # Classify operation type
        operation_type = self._classify_operation_type(content, function_name)

        return {
            "intent_category": primary_intent[0] if primary_intent[1] > 0 else "general",
            "intent_confidence": min(primary_intent[1] / 5.0, 1.0),  # Normalize to 0-1
            "tags": tags[:5],  # Top 5 relevant tags
            "operation_type": operation_type,
            "domain_area": self._classify_domain_area(file_path)
        }

    def _classify_operation_type(self, content: str, function_name: str) -> str:
        """Classify the type of operation this function performs."""
        content_lower = content.lower()
        name_lower = function_name.lower()

        if any(word in name_lower for word in ["create", "add", "insert", "new", "register"]):
            return "create"
        elif any(word in name_lower for word in ["get", "find", "retrieve", "fetch", "list", "search"]):
            return "read"
        elif any(word in name_lower for word in ["update", "modify", "edit", "change", "set"]):
            return "update"
        elif any(word in name_lower for word in ["delete", "remove", "destroy", "clear"]):
            return "delete"
        elif any(word in name_lower for word in ["validate", "verify", "check", "ensure"]):
            return "validation"
        elif any(word in name_lower for word in ["process", "handle", "execute", "run"]):
            return "processing"
        else:
            return "other"

    def _classify_domain_area(self, file_path: str) -> str:
        """Classify the domain/architectural area of the code."""
        path_lower = file_path.lower()

        domain_patterns = {
            "models": ["model", "entity", "schema", "data"],
            "controllers": ["controller", "handler", "endpoint", "route"],
            "services": ["service", "business", "logic", "manager"],
            "utils": ["util", "helper", "common", "shared"],
            "auth": ["auth", "login", "security", "permission"],
            "database": ["db", "database", "repository", "dao"],
            "frontend": ["view", "template", "ui", "frontend", "client"],
            "middleware": ["middleware", "filter", "interceptor"],
            "config": ["config", "setting", "env", "constant"],
            "test": ["test", "spec", "__test__", "tests"]
        }

        for domain, keywords in domain_patterns.items():
            if any(keyword in path_lower for keyword in keywords):
                return domain

        return "general"

    def _extract_quality_metadata(self, content: str, file_path: str, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract code quality and complexity indicators."""
        try:
            # Calculate cyclomatic complexity (simplified)
            complexity = self._calculate_complexity(node)

            # Check for error handling patterns
            has_error_handling = self._check_error_handling(content)

            # Check for logging
            has_logging = self._check_logging(content)

            # Check for documentation
            documentation_quality = self._assess_documentation_quality(node, content)

            # Extract error patterns
            error_patterns = self._extract_error_patterns(content)

            # Check for performance patterns
            performance_indicators = self._detect_performance_patterns(content)

            # Security patterns
            security_patterns = self._detect_security_patterns(content)

            return {
                "complexity_level": complexity,
                "has_error_handling": has_error_handling,
                "has_logging": has_logging,
                "documentation_quality": documentation_quality,
                "error_patterns": error_patterns,
                "performance_indicators": performance_indicators,
                "security_patterns": security_patterns,
                "code_smells": self._detect_code_smells(content),
                "maintainability_score": self._calculate_maintainability_score(
                    complexity, has_error_handling, has_logging, documentation_quality
                )
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract quality metadata: {e}")
            return {}

    def _calculate_complexity(self, node: ast.FunctionDef) -> str:
        """Calculate simplified cyclomatic complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        if complexity <= 5:
            return "low"
        elif complexity <= 10:
            return "medium"
        elif complexity <= 20:
            return "high"
        else:
            return "very_high"

    def _check_error_handling(self, content: str) -> bool:
        """Check if code has proper error handling."""
        error_patterns = ["try:", "except", "raise", "assert", "if not", "is None"]
        return any(pattern in content for pattern in error_patterns)

    def _check_logging(self, content: str) -> bool:
        """Check if code has logging statements."""
        log_patterns = ["logging", "logger", "log.", ".debug", ".info", ".warning", ".error", "print("]
        return any(pattern in content for pattern in log_patterns)

    def _assess_documentation_quality(self, node: ast.FunctionDef, content: str) -> str:
        """Assess the quality of documentation."""
        docstring = ast.get_docstring(node)

        if not docstring:
            return "none"
        elif len(docstring) < 20:
            return "minimal"
        elif len(docstring.split('\n')) >= 3 and any(word in docstring.lower() for word in ["args", "returns", "parameters"]):
            return "good"
        else:
            return "basic"

    def _extract_error_patterns(self, content: str) -> List[str]:
        """Extract specific error patterns and exception types."""
        patterns = []

        # Common exception types
        exceptions = ["ValueError", "TypeError", "KeyError", "AttributeError", "IndexError",
                     "FileNotFoundError", "ConnectionError", "TimeoutError", "ValidationError"]

        for exc in exceptions:
            if exc in content:
                patterns.append(exc)

        return patterns[:3]  # Top 3 most relevant

    def _detect_performance_patterns(self, content: str) -> List[str]:
        """Detect performance-related patterns."""
        patterns = []

        perf_indicators = {
            "database_query": ["SELECT", "INSERT", "UPDATE", "DELETE", "query", "execute"],
            "caching": ["cache", "redis", "memcache", "lru"],
            "async_operations": ["async", "await", "asyncio", "threading"],
            "file_operations": ["open(", "file", "read(", "write("],
            "network_requests": ["requests", "http", "api", "url"],
            "loops": ["for ", "while ", "map(", "filter("],
            "recursion": ["return self.", "return " + content.split("def ")[1].split("(")[0] if "def " in content else ""]
        }

        content_lower = content.lower()
        for pattern_type, indicators in perf_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                patterns.append(pattern_type)

        return patterns[:3]  # Top 3 most relevant

    def _detect_security_patterns(self, content: str) -> List[str]:
        """Detect security-related patterns."""
        patterns = []

        security_indicators = {
            "authentication": ["login", "auth", "authenticate", "verify", "credential"],
            "authorization": ["permission", "access", "role", "grant", "deny"],
            "encryption": ["encrypt", "decrypt", "hash", "bcrypt", "sha", "md5"],
            "validation": ["validate", "sanitize", "escape", "filter"],
            "input_handling": ["request.", "input", "form", "param"],
            "password_handling": ["password", "passwd", "pwd", "secret"],
            "token_handling": ["token", "jwt", "bearer", "session"],
            "sql_injection": ["sql", "query", "execute", "raw"]
        }

        content_lower = content.lower()
        for pattern_type, indicators in security_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                patterns.append(pattern_type)

        return patterns[:3]  # Top 3 most relevant

    def _detect_code_smells(self, content: str) -> List[str]:
        """Detect common code smells."""
        smells = []

        # Long function (simplified check)
        line_count = len(content.split('\n'))
        if line_count > 50:
            smells.append("long_function")

        # Too many parameters (simplified check)
        if content.count(',') > 5 and 'def ' in content:
            smells.append("too_many_parameters")

        # Duplicate code patterns
        if content.count('if ') > 5:
            smells.append("complex_conditionals")

        # Magic numbers
        if re.search(r'\b\d{2,}\b', content):
            smells.append("magic_numbers")

        return smells[:3]  # Top 3 most relevant

    def _calculate_maintainability_score(self, complexity: str, has_error_handling: bool,
                                       has_logging: bool, documentation_quality: str) -> float:
        """Calculate a maintainability score (0-1)."""
        score = 1.0

        # Complexity penalty
        complexity_penalty = {
            "low": 0.0,
            "medium": 0.1,
            "high": 0.3,
            "very_high": 0.5
        }
        score -= complexity_penalty.get(complexity, 0.2)

        # Error handling bonus
        if has_error_handling:
            score += 0.1
        else:
            score -= 0.2

        # Logging bonus
        if has_logging:
            score += 0.05

        # Documentation bonus
        doc_bonus = {
            "none": -0.2,
            "minimal": -0.1,
            "basic": 0.0,
            "good": 0.15
        }
        score += doc_bonus.get(documentation_quality, 0.0)

        return max(0.0, min(1.0, score))  # Clamp to 0-1