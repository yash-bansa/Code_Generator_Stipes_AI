import json
import ast
import re
import os
import logging
import time
from datetime import datetime
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from utils.llm_client import llm_client
from utils.file_handler import FileHandler
from config.agents_io import (
    CodeValidatorInput, 
    CodeValidatorOutput, 
    FileValidationResult, 
    ValidationSummary, 
    CodeMetrics,
    FileToValidate
)

logger = logging.getLogger(__name__)

class CodeValidatorAgent:
    def __init__(self):
        """Initialize the Code Validator Agent with configuration"""
        config_path = Path(__file__).parent / "config" / "code_validator_config.yaml"
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.system_prompt = self.config["system_prompt"]
        except Exception as e:
            logger.error(f"[CodeValidatorAgent] Failed to load config: {e}")

        # Initialize validation rules and patterns in code
        self._initialize_validation_rules()

    def _initialize_validation_rules(self):
        """Initialize validation rules and patterns"""
        self.validation_rules = {
            "syntax_check": True,
            "security_scan": True, 
            "performance_analysis": True,
            "style_check": True,
            "complexity_analysis": True,
            "documentation_check": True
        }

        self.security_patterns = {
            "dangerous_functions": [
                {"pattern": r'eval\s*\(', "message": "Use of eval() detected - security risk", "severity": "high"},
                {"pattern": r'exec\s*\(', "message": "Use of exec() detected - security risk", "severity": "high"},
                {"pattern": r'input\s*\(', "message": "Use of input() - validate inputs", "severity": "medium"},
                {"pattern": r'subprocess\.(call|run|Popen)', "message": "Use of subprocess - ensure safe commands", "severity": "medium"},
                {"pattern": r'os\.system\s*\(', "message": "Use of os.system() - injection risk", "severity": "high"},
                {"pattern": r'pickle\.loads?\s*\(', "message": "Use of pickle - untrusted input risk", "severity": "medium"},
                {"pattern": r'__import__\s*\(', "message": "Dynamic import - validate input", "severity": "low"}
            ],
            "secret_patterns": [
                {"pattern": r'password\s*=\s*["\'][^"\']+["\']', "message": "Hardcoded password detected", "severity": "high"},
                {"pattern": r'api_key\s*=\s*["\'][^"\']+["\']', "message": "Hardcoded API key detected", "severity": "high"},
                {"pattern": r'secret\s*=\s*["\'][^"\']+["\']', "message": "Hardcoded secret detected", "severity": "high"},
                {"pattern": r'token\s*=\s*["\'][^"\']+["\']', "message": "Hardcoded token detected", "severity": "high"}
            ]
        }

        self.performance_patterns = [
            {"pattern": r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', "message": "Use enumerate() instead of range(len())", "category": "iteration"},
            {"pattern": r'\+=.*str', "message": "Use join() for string concatenation in loops", "category": "string_ops"},
            {"pattern": r'\.append\s*\([^)]*for\s+', "message": "Use list comprehension instead of appending in loop", "category": "list_ops"},
            {"pattern": r'global\s+', "message": "Avoid global variables for performance and clarity", "category": "scope"}
        ]

        self.complexity_thresholds = {
            "function_length_warning": 50,
            "function_length_error": 100,
            "complexity_low": 10,
            "complexity_medium": 20,
            "complexity_high": 30
        }

        self.quality_scoring = {
            "base_score": 100,
            "error_penalty": -10,
            "warning_penalty": -2,
            "missing_docstring_penalty": -1,
            "high_complexity_penalty": -5,
            "security_issue_penalty": -15
        }

    async def validate_code_changes(self, input_data: CodeValidatorInput) -> CodeValidatorOutput:
        """Main method to validate code modifications"""
        start_time = time.time()
        
        logger.info(f"[CodeValidatorAgent] Starting validation for {len(input_data.modified_files)} files")
        
        # Initialize output structure
        files_validated = []
        all_errors = []
        all_warnings = []
        all_suggestions = []
        
        # Process each file
        for file_info in input_data.modified_files:
            try:
                file_validation = await self._validate_single_file(
                    file_info.file_path, 
                    file_info.modified_content,
                    input_data.strict_mode
                )
                files_validated.append(file_validation)
                
                # Collect all issues
                all_errors.extend(file_validation.errors)
                if not input_data.skip_warnings:
                    all_warnings.extend(file_validation.warnings)
                all_suggestions.extend(file_validation.suggestions)
                
            except Exception as e:
                logger.exception(f"[CodeValidatorAgent] Error validating file {file_info.file_path}")
                # Create error result for failed validation
                error_result = FileValidationResult(
                    file_path=file_info.file_path,
                    syntax_valid=False,
                    errors=[f"Validation failed: {str(e)}"],
                    warnings=[],
                    suggestions=[],
                    metrics=CodeMetrics(
                        lines_of_code=0, blank_lines=0, comment_lines=0,
                        functions_count=0, classes_count=0, imports_count=0,
                        complexity_estimate="unknown", complexity_score=0
                    ),
                    validation_passed=False
                )
                files_validated.append(error_result)
                all_errors.append(f"{file_info.file_path}: {str(e)}")

        # Create validation summary
        validation_summary = self._create_validation_summary(files_validated)
        
        # Determine overall status
        overall_status = self._determine_overall_status(validation_summary, input_data.strict_mode)
        
        # Create output
        output = CodeValidatorOutput(
            success=True,
            overall_status=overall_status,
            files_validated=files_validated,
            validation_summary=validation_summary,
            errors_found=all_errors,
            warnings=all_warnings,
            suggestions=all_suggestions,
            execution_time=time.time() - start_time,
            timestamp=datetime.now()
        )
        
        logger.info(f"[CodeValidatorAgent] Validation completed: {overall_status} - {validation_summary.total_files} files processed")
        
        return output

    async def _validate_single_file(self, file_path: str, content: str, strict_mode: bool = False) -> FileValidationResult:
        """Validate a single file and return detailed results"""
        errors = []
        warnings = []
        suggestions = []
        syntax_valid = True
        
        # Syntax validation
        if self.validation_rules.get("syntax_check", True):
            is_valid, syntax_error = FileHandler.validate_python_syntax(content)
            if not is_valid:
                syntax_valid = False
                errors.append(f"Syntax Error: {syntax_error}")
        
        # Only proceed with other checks if syntax is valid
        if syntax_valid:
            try:
                # Static analysis
                static_issues = self._perform_static_analysis(content)
                errors.extend(static_issues["errors"])
                warnings.extend(static_issues["warnings"])
                suggestions.extend(static_issues["suggestions"])
                
                # Security analysis
                if self.validation_rules.get("security_scan", True):
                    security_issues = self._check_security_issues(content)
                    if strict_mode:
                        errors.extend(security_issues)
                    else:
                        warnings.extend(security_issues)
                
                # Performance analysis
                if self.validation_rules.get("performance_analysis", True):
                    perf_suggestions = self._analyze_performance(content)
                    suggestions.extend(perf_suggestions)
                
            except Exception as e:
                errors.append(f"Analysis failed: {e}")
        
        # Calculate metrics
        metrics = self._calculate_code_metrics(content)
        
        # Determine if validation passed
        validation_passed = len(errors) == 0 and (not strict_mode or len(warnings) == 0)
        
        return FileValidationResult(
            file_path=file_path,
            syntax_valid=syntax_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics,
            validation_passed=validation_passed
        )

    def _perform_static_analysis(self, content: str) -> Dict[str, List[str]]:
        """Perform static code analysis"""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for bare except clauses
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues["warnings"].append(f"Line {node.lineno}: Bare except clause detected")
                
                # Check for unused variables (starting with _)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.startswith('_') and not target.id.startswith('__'):
                            issues["suggestions"].append(f"Line {node.lineno}: Variable {target.id} appears unused")
                
                # Check function length
                if isinstance(node, ast.FunctionDef):
                    if hasattr(node, 'end_lineno'):
                        length = node.end_lineno - node.lineno
                        warning_threshold = self.complexity_thresholds.get("function_length_warning", 50)
                        error_threshold = self.complexity_thresholds.get("function_length_error", 100)
                        
                        if length > error_threshold:
                            issues["errors"].append(f"Line {node.lineno}: Function '{node.name}' is too long ({length} lines)")
                        elif length > warning_threshold:
                            issues["warnings"].append(f"Line {node.lineno}: Function '{node.name}' is very long ({length} lines)")
                
                # Check for missing docstrings
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                        issues["warnings"].append(f"Line {node.lineno}: {node.__class__.__name__} '{node.name}' missing docstring")
        
        except Exception as e:
            issues["errors"].append(f"Static analysis error: {e}")
        
        return issues

    def _calculate_code_metrics(self, content: str) -> CodeMetrics:
        """Calculate detailed code quality metrics"""
        lines = content.splitlines()
        complexity_keywords = ['if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except:', 'with ']
        complexity_score = sum(content.count(k) for k in complexity_keywords)
        
        # Determine complexity level
        low_threshold = self.complexity_thresholds.get("complexity_low", 10)
        medium_threshold = self.complexity_thresholds.get("complexity_medium", 20)
        
        if complexity_score > medium_threshold:
            complexity_estimate = "high"
        elif complexity_score > low_threshold:
            complexity_estimate = "medium"
        else:
            complexity_estimate = "low"
        
        return CodeMetrics(
            lines_of_code=len(lines),
            blank_lines=len([l for l in lines if not l.strip()]),
            comment_lines=len([l for l in lines if l.strip().startswith('#')]),
            functions_count=content.count('def '),
            classes_count=content.count('class '),
            imports_count=content.count('import ') + content.count('from '),
            complexity_estimate=complexity_estimate,
            complexity_score=complexity_score
        )

    def _check_security_issues(self, content: str) -> List[str]:
        """Check for security vulnerabilities"""
        issues = []
        
        # Check dangerous functions
        for pattern_info in self.security_patterns.get("dangerous_functions", []):
            pattern = pattern_info.get("pattern", "")
            message = pattern_info.get("message", "Security issue detected")
            if pattern and re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Security Warning: {message}")
        
        # Check for hardcoded secrets
        for pattern_info in self.security_patterns.get("secret_patterns", []):
            pattern = pattern_info.get("pattern", "")
            message = pattern_info.get("message", "Hardcoded secret detected")
            if pattern and re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Security Warning: {message}")
        
        return issues

    def _analyze_performance(self, content: str) -> List[str]:
        """Analyze code for performance issues"""
        suggestions = []
        
        for pattern_info in self.performance_patterns:
            pattern = pattern_info.get("pattern", "")
            message = pattern_info.get("message", "Performance improvement available")
            if pattern and re.search(pattern, content):
                suggestions.append(f"Performance: {message}")
        
        return suggestions

    def _create_validation_summary(self, files_validated: List[FileValidationResult]) -> ValidationSummary:
        """Create a summary of validation results"""
        total_files = len(files_validated)
        files_with_errors = sum(1 for f in files_validated if f.errors)
        files_with_warnings = sum(1 for f in files_validated if f.warnings)
        files_passed = sum(1 for f in files_validated if f.validation_passed)
        total_errors = sum(len(f.errors) for f in files_validated)
        total_warnings = sum(len(f.warnings) for f in files_validated)
        total_suggestions = sum(len(f.suggestions) for f in files_validated)
        
        # Calculate overall quality score
        base_score = self.quality_scoring.get("base_score", 100)
        error_penalty = self.quality_scoring.get("error_penalty", -10)
        warning_penalty = self.quality_scoring.get("warning_penalty", -2)
        
        quality_score = base_score + (total_errors * error_penalty) + (total_warnings * warning_penalty)
        quality_score = max(0, min(100, quality_score))  # Clamp between 0-100
        
        return ValidationSummary(
            total_files=total_files,
            files_with_errors=files_with_errors,
            files_with_warnings=files_with_warnings,
            files_passed=files_passed,
            total_errors=total_errors,
            total_warnings=total_warnings,
            total_suggestions=total_suggestions,
            overall_quality_score=quality_score
        )

    def _determine_overall_status(self, summary: ValidationSummary, strict_mode: bool) -> str:
        """Determine the overall validation status"""
        if summary.total_errors > 0:
            return "failed"
        elif strict_mode and summary.total_warnings > 0:
            return "failed"
        elif summary.total_warnings > 0:
            return "warning"
        else:
            return "passed"