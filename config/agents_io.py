from pydantic import BaseModel, Field
from typing import Dict,Any,List, Optional, Literal, Union
from pathlib import Path
from datetime import datetime




# -------------------------------
# CommunicationAgent Contracts
# -------------------------------

class CommunicationInput(BaseModel):
    user_query: str
    conversation_history: List[str]

class CommunicationOutput(BaseModel):
    core_intent: str
    context_notes: str
    success: bool = True
    message: str = "Intent extracted successfully"

# -------------------------------
# QueryRephraserAgent Contracts
# -------------------------------

class QueryEnhancerInput(BaseModel):
    core_intent: str
    context_notes: Optional[str] = None

class QueryEnhancerOutput(BaseModel):
    developer_task: str
    is_satisfied: bool
    suggestions: List[str] = Field(default_factory=list)
    success: bool = True
    message: str = "Query enhanced successfully"
    reason :str
    change_type :str


# -------------------------------
# Master Planner Agent Contracts
# -------------------------------


class MasterPlannerInput(BaseModel):
    parsed_config: Dict[str, Any]
    # project_path: Path
    user_question: str


class FileAnalysisResult(BaseModel):
    needs_modification: bool
    modification_type: Optional[
        Literal["data_loading", "data_transformation", "output_handling", "configuration", "utility"]
    ] = ""
    priority: Literal["high", "medium", "low"] = "low"
    reason: str
    # suggested_changes: List[SuggestedChange] = Field(default_factory=list)
    # cross_file_dependencies: List[CrossFileDependency] = Field(default_factory=list)


class TargetFileOutput(BaseModel):
    file_path: str
    file_info: Dict[str, Any]
    # structure: Dict[str, Any]
    analysis: FileAnalysisResult
    priority: Literal["high", "medium", "low"]


class MasterPlannerOutput(BaseModel):
    success: bool
    message: str
    files_to_modify: List[TargetFileOutput] = Field(default_factory=list)

# -------------------------------
# Delta Analyzer Agent Contracts
# -------------------------------

class Modification(BaseModel):
    action: str
    target_type: str
    target_name: str
    line_number: Optional[int] = 0
    old_code: Optional[str] = ""
    new_code: str
    explanation: str
    affects_dependencies: Optional[List[str]] = Field(default_factory=list)  # KEEP
    user_intent_alignment : Optional[float] = 0.0

class DeltaAnalyzerInput(BaseModel):
    file_path: str
    file_content: str
    file_analysis: Dict[str, Any] = Field(default_factory=dict) # Contains cross_dependencies already
    config: Dict[str, Any] = Field(default_factory=dict)
    user_query : str = ""
    # NO additional fields needed - keep it clean

class DeltaAnalyzerOutput(BaseModel):
    modifications: List[Modification] = Field(default_factory=list) 
    new_dependencies: List[str] = Field(default_factory=list)
    testing_suggestions: List[str] = Field(default_factory=list)
    potential_issues: List[str] = Field(default_factory=list)
    cross_file_impacts: Optional[List[str]] = Field(default_factory=list)      # KEEP
    implementation_notes: Optional[List[str]] = Field(default_factory=list)    # KEEP
    user_query_analysis : Dict[str, Any] = Field(default_factory=dict)
    alignment_score :Optional[float] = 0.0
    original_file_content : Optional[str] = Field(default="")

###### Code Generator Agent

# Code Generator Models
class ModificationItem(BaseModel):
    """Individual code modification from DeltaAnalyzer"""
    action: str = Field(..., description="Type of action: add, modify, delete")
    target_type: str = Field(..., description="Type of target: function, class, import, variable, comment")
    target_name: str = Field(..., description="Name of the target element")
    line_number: int = Field(default=0, description="Line number for modification")
    old_code: Optional[str] = Field(None, description="Existing code to replace")
    new_code: Optional[str] = Field(None, description="New or modified code")  # Agent handles None case
    explanation: str = Field(..., description="Explanation of the modification")
    affects_dependencies: List[str] = Field(default_factory=list, description="Files that might be affected")
    user_intent_alignment: float = Field(default=0.8, description="Alignment with user intent")

class FileModificationPlan(BaseModel):
    """Plan for modifying a single file"""
    file_path: str = Field(..., description="Path to the file")
    priority: str = Field(default="medium", description="Priority level")
    modification_type: str = Field(default="general", description="Type of modification")
    modifications: List[ModificationItem] = Field(default_factory=list, description="List of modifications")
    new_dependencies: List[str] = Field(default_factory=list, description="New dependencies needed")
    testing_suggestions: List[str] = Field(default_factory=list, description="Testing recommendations")
    potential_issues: List[str] = Field(default_factory=list, description="Potential issues")
    user_alignment_score: float = Field(default=0.5, description="Alignment with user query")

class CodeGeneratorInput(BaseModel):
    """Input for Code Generator Agent"""
    modification_plan: Dict[str, Any] = Field(..., description="Plan from DeltaAnalyzer")
    user_query: str = Field(default="", description="Original user query")  # Agent uses input_data.user_query

class GeneratedFile(BaseModel):
    """Information about a generated/modified file"""
    file_path: str = Field(..., description="Path to the file")
    original_content: Optional[str] = Field(None, description="Original file content")  # Keep Optional (agent sets to current_content)
    modified_content: str = Field(..., description="Modified file content")
    modifications_applied: int = Field(..., description="Number of modifications applied")
    backup_path: Optional[str] = Field(None, description="Path to backup file")  # Agent sets to None

class CodeGeneratorOutput(BaseModel):
    """Output from Code Generator Agent"""
    success: bool = Field(..., description="Overall success status")
    modified_files: List[GeneratedFile] = Field(default_factory=list, description="Successfully modified files")
    failed_files: List[str] = Field(default_factory=list, description="Files that failed to modify")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated")
    total_modifications: int = Field(default=0, description="Total number of modifications made")
    execution_time: float = Field(default=0.0, description="Time taken for execution")


#### code validator

class FileToValidate(BaseModel):
    file_path: str = Field(..., description="Path to the file being validated")
    original_content: str = Field(..., description="Original file content before modifications")
    modified_content: str = Field(..., description="Modified file content to be validated")
    modifications_applied: int = Field(..., description="Number of modifications applied to this file")
    backup_path: Optional[str] = Field(None, description="Path to backup file if exists")

class CodeValidatorInput(BaseModel):
    modified_files: List[FileToValidate] = Field(..., description="List of files to validate")
    validation_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional validation configuration options"
    )
    strict_mode: bool = Field(
        default=False,
        description="Whether to use strict validation rules"
    )
    skip_warnings: bool = Field(
        default=False,
        description="Whether to skip warning-level issues"
    )

# Output Models
class CodeMetrics(BaseModel):
    lines_of_code: int = Field(..., description="Total lines of code")
    blank_lines: int = Field(..., description="Number of blank lines")
    comment_lines: int = Field(..., description="Number of comment lines")
    functions_count: int = Field(..., description="Number of functions")
    classes_count: int = Field(..., description="Number of classes")
    imports_count: int = Field(..., description="Number of import statements")
    complexity_estimate: str = Field(..., description="Complexity level: low/medium/high")
    complexity_score: int = Field(..., description="Numerical complexity score")

class FileValidationResult(BaseModel):
    file_path: str = Field(..., description="Path of the validated file")
    syntax_valid: bool = Field(..., description="Whether the file has valid Python syntax")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    warnings: List[str] = Field(default_factory=list, description="List of warning messages")
    suggestions: List[str] = Field(default_factory=list, description="List of improvement suggestions")
    metrics: CodeMetrics = Field(..., description="Code quality metrics")
    validation_passed: bool = Field(..., description="Overall validation status for this file")

class ValidationSummary(BaseModel):
    total_files: int = Field(..., description="Total number of files validated")
    files_with_errors: int = Field(..., description="Number of files with errors")
    files_with_warnings: int = Field(..., description="Number of files with warnings")
    files_passed: int = Field(..., description="Number of files that passed validation")
    total_errors: int = Field(..., description="Total number of errors found")
    total_warnings: int = Field(..., description="Total number of warnings found")
    total_suggestions: int = Field(..., description="Total number of suggestions made")
    overall_quality_score: float = Field(..., description="Overall code quality score (0-100)")

class CodeValidatorOutput(BaseModel):
    success: bool = Field(..., description="Whether the validation process completed successfully")
    overall_status: str = Field(..., description="Overall validation status: passed/failed/warning")
    files_validated: List[FileValidationResult] = Field(
        default_factory=list,
        description="Detailed validation results for each file"
    )
    validation_summary: ValidationSummary = Field(..., description="Summary of validation results")
    errors_found: List[str] = Field(default_factory=list, description="All errors found across files")
    warnings: List[str] = Field(default_factory=list, description="All warnings found across files")
    suggestions: List[str] = Field(default_factory=list, description="All suggestions across files")
    execution_time: float = Field(..., description="Time taken for validation in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When validation was performed")

class BotStateSchema(BaseModel):
    latest_query: str
    user_history: List[str]
    current_user: Optional[str] = "default_user"  # NEW: Add current user field

    # Communication Agent Output
    core_intent: Optional[str] = ""
    context_notes: Optional[str] = ""
    communication_success: bool = False

    # Query Rephraser Agent Output
    developer_task: Optional[str] = ""
    is_satisfied: bool = False
    suggestions: List[str] = []
    enhancement_success: bool = False

    # Master Planner Agent Output
    master_planner_success: bool = False
    master_planner_message: Optional[str] = ""
    master_planner_result: List[TargetFileOutput] = Field(default_factory=list)
    parsed_config: Optional[Dict[str, Any]] = Field(default_factory=dict)  # Store parsed config for Delta Analyzer

    # Delta Analyzer Agent Output
    delta_analyzer_success: bool = False
    delta_analyzer_message: Optional[str] = ""
    delta_analyzer_result: Optional[DeltaAnalyzerOutput] = None  # UPDATED: Store complete Delta Analyzer output
    modification_plan: Optional[Dict[str, Any]] = Field(default_factory=dict)  # Keep for backward compatibility

    # Code Generator Agent Output - NEW SECTION
    code_generator_success: bool = False
    code_generator_message: Optional[str] = ""
    code_generator_result: Optional[CodeGeneratorOutput] = None

    # Code Validator Agent Output - NEW SECTION
    code_validator_success: bool = False
    code_validator_message: Optional[str] = ""
    code_validator_result: Optional[CodeValidatorOutput] = None