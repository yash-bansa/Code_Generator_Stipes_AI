import asyncio
import sys
import json
import logging
import threading
from threading import Lock
from pathlib import Path
from datetime import datetime
from langgraph.graph import StateGraph
from typing import List, Union, Dict, Any
from pydantic import BaseModel
from agents import *
from config.agents_io import *
from utils.file_handler import FileHandler
from config.settings import settings
from dotenv import load_dotenv
from langfuse import Langfuse
import os

load_dotenv()

# ---------- Thread Safety ----------
graph_lock = Lock()

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("langraph_query_flow.log")
    ]
)
logger = logging.getLogger(__name__)

# ---------- Redis Connection ----------
redis_client = settings.get_redis_connection()

# ---------- Agent Initialization ----------
print("\nInitializing LangGraph-style Query Agents...")
communication_agent = CommunicationAgent()
query_rephraser_agent = QueryRephraserAgent()
master_planner_agent = MasterPlannerAgent()
delta_analyzer_agent = DeltaAnalyzerAgent()
code_generator_agent = CodeGeneratorAgent()
code_validator_agent = CodeValidatorAgent()  # NEW: Code Validator Agent
config_generator_agent = DependencyExtractorAgent()
print("Agents initialized successfully!")

def ensure_state_schema(state: Union[dict, BaseModel]) -> BotStateSchema:
    if isinstance(state, BotStateSchema):
        return state
    return BotStateSchema(**state)

def convert_delta_to_generator_format(delta_result: DeltaAnalyzerOutput, master_planner_result: List) -> dict:
    """Convert Delta Analyzer output to Code Generator input format"""
    files_to_modify = []

    # Group modifications by file path (we need to extract this from modifications)
    file_modifications = {}

    for mod in delta_result.modifications:
        # We need to determine which file this modification belongs to
        # This should ideally come from the modification itself
        file_path = getattr(mod, 'file_path', None)

        # If no file_path in modification, try to match with master planner results
        if not file_path and master_planner_result:
            # For now, assign to first file - this should be improved based on your schema
            file_path = master_planner_result[0].file_path if master_planner_result else "unknown.py"

        if file_path not in file_modifications:
            file_modifications[file_path] = {
                "file_path": file_path,
                "suggestions": {
                    "original_file_content": "",  # This should come from somewhere
                    "modifications": [],
                    "new_dependencies": [],
                    "testing_suggestions": [],
                    "potential_issues": [],
                    "cross_file_impacts": [],
                    "implementation_notes": []
                }
            }

        # Add modification to the file
        file_modifications[file_path]["suggestions"]["modifications"].append({
            "action": mod.action,
            "target_type": mod.target_type,
            "target_name": mod.target_name,
            "line_number": mod.line_number,
            "old_code": mod.old_code,
            "new_code": mod.new_code,
            "explanation": mod.explanation,
            "affects_dependencies": mod.affects_dependencies
        })

    # Add global suggestions to all files
    for file_path in file_modifications:
        file_modifications[file_path]["suggestions"]["new_dependencies"] = delta_result.new_dependencies
        file_modifications[file_path]["suggestions"]["testing_suggestions"] = delta_result.testing_suggestions
        file_modifications[file_path]["suggestions"]["potential_issues"] = delta_result.potential_issues
        file_modifications[file_path]["suggestions"]["cross_file_impacts"] = delta_result.cross_file_impacts
        file_modifications[file_path]["suggestions"]["implementation_notes"] = delta_result.implementation_notes

    return {
        "files_to_modify": list(file_modifications.values())
    }

# NEW: Function to convert Code Generator output to Code Validator input
def convert_generator_to_validator_format(generator_output: CodeGeneratorOutput) -> CodeValidatorInput:
    """Convert Code Generator output to Code Validator input format"""
    files_to_validate = []

    for modified_file in generator_output.modified_files:
        file_to_validate = FileToValidate(
            file_path=modified_file.file_path,
            original_content=modified_file.original_content,
            modified_content=modified_file.modified_content,
            modifications_applied=modified_file.modifications_applied,
            backup_path=modified_file.backup_path
        )
        files_to_validate.append(file_to_validate)

    return CodeValidatorInput(
        modified_files=files_to_validate,
        strict_mode=True,  # Use strict mode for thorough validation
        skip_warnings=False  # Don't skip warnings
    )

def get_or_create_trace(session_id: str):

    langfuse = Langfuse(
        secret_key = os.getenv("LANGFUSE_DEV_SECRET_KEY"),
        public_key = os.getenv("LANGFUSE_DEV_PUBLIC_KEY"),
        host = os.getenv("LANGFUSE_DEV_HOST")
    )

    trace = langfuse.trace(
        id = session_id,
        session_id = session_id,
        name = f"multi_agent_workflow_{session_id}",
        environment="test_env"
    )
    return trace


# ---------- LangGraph-Compatible Nodes ----------
async def communication_node(state: dict) -> dict:
    session_id = state["current_user"]
    print("I am inside Communication Node")

    state_obj = ensure_state_schema(state)
    logger.info("Communication Node: Extracting intent...")

    trace = get_or_create_trace(session_id)

    comm_input = CommunicationInput(
        user_query=state_obj.latest_query,
        conversation_history=state_obj.user_history[:-1] if len(state_obj.user_history) > 1 else []
    )

    generation = trace.generation(
        name="communication_agent_step",
        model="gpt-4o",
        input=comm_input.model_dump()
    )

    result: CommunicationOutput = await communication_agent.extract_intent(comm_input)

    generation.update(output=result.model_dump())
    generation.end()

    state_obj.core_intent = result.core_intent
    state_obj.context_notes = result.context_notes
    state_obj.communication_success = result.success

    logger.info(f"Core Intent: {result.core_intent}")
    logger.info(f"Context Notes: {result.context_notes}")

    return state_obj.dict()

async def query_enhancement_node(state: dict) -> dict:

    state_obj = ensure_state_schema(state)
    if state['master_planner_approved'] == True:
        print("The master planner has been approved, so skipping query enhancer")
        return state_obj.dict()


    session_id = state["current_user"]
    print("I am inside query rephrase agent")

    logger.info("Query Enhancement Node: Rephrasing and validating...")

    trace = get_or_create_trace(session_id)

    enhancer_input = QueryEnhancerInput(
        core_intent=state_obj.core_intent,
        context_notes=state_obj.context_notes
    )

    generation = trace.generation(
        name="query_rephrase_agent_step",
        model="gpt-4o",
        input=enhancer_input.model_dump()
    )
    result: QueryEnhancerOutput = await query_rephraser_agent.enhance_query(enhancer_input)
    generation.update(output=result.model_dump())
    generation.end()

    state_obj.developer_task = result.developer_task
    state_obj.is_satisfied = result.is_satisfied
    state_obj.suggestions = result.suggestions
    state_obj.enhancement_success = result.success
    state_obj.change_type = result.change_type

    print("I an Inside the query enhancement node")
    logger.info(f"Change Type: {result.change_type}")
    logger.info(f"Developer Task: {result.developer_task}")
    logger.info(f"Is Satisfied: {result.is_satisfied}")
    if not result.is_satisfied:
        logger.info("Suggestions:")
        for s in result.suggestions:
            logger.info(f"- {s}")

    return state_obj.dict()

async def master_planner_node(state: dict) -> dict:
    state_obj = ensure_state_schema(state)
    if state['master_planner_approved'] == True:
        print("The master planner has been approved, so skipping master planner")
        return state_obj.dict()

    session_id = state["current_user"]
    trace = get_or_create_trace(session_id)

    logger.info("Master Planner Node: Identifying target files...")

    try:
        if state_obj.change_type == "config_change":
            if not state_obj.updated_config:
                tables_list = await config_generator_agent.extract_table_names_from_query(state_obj.developer_task)
                final_tables = await config_generator_agent.filter_and_format_table_paths(tables_list)
                print("table_list", tables_list)
                print("final_tables", final_tables)
                updated_result = []
                for table in final_tables:
                    val = {}
                    config = await config_generator_agent.read_config_file(table)

                    dependency_input = DependencyAnalyzerInput(
                        user_query=state_obj.developer_task,
                        config=config
                    )

                    result = await config_generator_agent.analyze_dependencies(dependency_input)


                    generation = trace.generation(
                    name="dependency_extractor_agent_step",
                    model="gpt-4o",
                    input=dependency_input.model_dump()
                    )
                    generation.update(output=result)
                    generation.end()

                    val["file_path"] = table
                    val["updated_config"] = result
                    updated_result.append(val)

                print("I am inside master planner config node")
                logger.info(f"Updated config: {updated_result}")
                state_obj.updated_config = str(updated_result)
                req_rag_output = str(updated_result)
            else:
                req_rag_output = state_obj.updated_config
        else:
            print("I am inside master planner code generator node")
            result = await master_planner_agent.detect_migration_with_llm(state_obj.developer_task)
            test_mig = result["is_migration"]
            print("Is Migration" , test_mig)
            if test_mig:
                print("Inside the migration node")
                repo_mig = await master_planner_agent.detect_migration_type_with_llm(state_obj.developer_task)
                print("whate type of migration", repo_mig)
                if repo_mig:
                    files = []
                    repos = await master_planner_agent.extract_repos_from_query(state_obj.developer_task)
                    for repo in repos:
                        files.extend(await FileHandler._scan_repository(repo))
                    files_dict = [{"file_path" : ele} for ele in files]
                    req_rag_output = str(files_dict) + "\n here you have the list of all files which are present in the code repo i want you to consider each file for thr code migration purpose or other type of full repo update and based on the user query set the expectation for each file clearly and based on the user query set the expectations for each files clearly and include all the files."
                else:
                    input_data = DocumentGeneratorInput(
                        developer_task_query=state_obj.developer_task
                    )
                    Document = DocumentGeneratorAgent()
                    rag_output = await Document.generate_document(input_data)
                    req_rag_output = rag_output.generated_doc

                    generation = trace.generation(
                    name="document_generator_agent_step",
                    model="gpt-4o",
                    input=input_data.model_dump()
                    )
                    generation.update(output=rag_output.model_dump())
                    generation.end()

            else:
                print("simple code generation node")
                input_data = DocumentGeneratorInput(
                        developer_task_query=state_obj.developer_task
                    )

                Document = DocumentGeneratorAgent()
                rag_output = await Document.generate_document(input_data)
                req_rag_output = rag_output.generated_doc


                generation = trace.generation(
                name="document_generator_agent_step",
                model="gpt-4o",
                input=input_data.model_dump()
                )
                generation.update(output=rag_output.model_dump())
                generation.end()

        if req_rag_output is None:
            print("did not get rag output")

        config_path = Path("./examples/sample_config.json")
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}. Creating default config...")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config = {
                "project_type": "python",
                "framework": "general",
                "main_files": ["main.py", "app.py"],
                "config_files": ["config.py", "settings.py"]
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config at: {config_path}")

        with open(config_path, 'r') as f:
            parsed_config = json.load(f)

        state_obj.parsed_config = parsed_config


        planner_input = MasterPlannerInput(
            parsed_config=parsed_config,
            user_question=state_obj.developer_task
        )


        result: MasterPlannerOutput = await master_planner_agent.identify_target_files(planner_input, rag_result=req_rag_output)


        generation = trace.generation(
        name="master_planner_agent_step",
        model="gpt-4o",
        input=planner_input.model_dump()
        )
        generation.update(output=result.model_dump())
        generation.end()

        state_obj.master_planner_result = result.files_to_modify
        state_obj.master_planner_success = result.success
        state_obj.master_planner_message = result.message

        logger.info(f"Master Planner Success: {result.success}")
        logger.info(f"Master Planner Message: {result.message}")
        logger.info(f"Files to Modify: {len(result.files_to_modify)}")

    except Exception as e:
        logger.error(f"Error in Master Planner Node: {e}")
        state_obj.master_planner_success = False
        state_obj.master_planner_message = f"Error: {str(e)}"
        state_obj.master_planner_result = []

    return state_obj.dict()

async def delta_analyzer_node(state: dict) -> dict:
    session_id = state["current_user"]
    trace = get_or_create_trace(session_id)
    state_obj = ensure_state_schema(state)
    logger.info("Delta Analyzer Node: Creating modification plan...")

    try:
        if state_obj.change_type == "code_change":

            target_files = state_obj.master_planner_result
            parsed_config = state_obj.parsed_config
            user_query = state_obj.developer_task

            delta_analyzer_input_dict = {
                "target_files" : target_files,
                "parsed_config": parsed_config,
                "user_query" : user_query
            }

            if not target_files:
                logger.warning("No target files available from Master Planner")
                state_obj.delta_analyzer_success = False
                state_obj.delta_analyzer_message = "No target files available from Master Planner"
                state_obj.delta_analyzer_result = None
                return state_obj.dict()

            final_target_files = []
            for target_file in target_files:
                if target_file.analysis.needs_modification == True:
                    final_target_files.append(target_file)

            target_files = final_target_files
            result_delta = await delta_analyzer_agent.create_modification_plan(target_files,parsed_config,user_query)

            generation = trace.generation(
            name="delta_analyzer_agent_step",
            model="gpt-4o",
            input=delta_analyzer_input_dict
            )
            generation.update(output=result_delta)
            generation.end()

        else:
            result_delta = "its a config change"
            generation = trace.generation(
            name="delta_analyzer_agent_step",
            model="gpt-4o",
            input="Delta analyzer agent is not required in config change."
            )
            generation.update(output=result_delta)
            generation.end()


        state_obj.delta_analyzer_result = result_delta
        state_obj.delta_analyzer_success = True
        state_obj.delta_analyzer_message = "Delta Analyzer completed successfully"

        logger.info(f"Delta Analyzer Success: True")

    except Exception as e:
        logger.error(f"Error in Delta Analyzer Node: {e}")
        state_obj.delta_analyzer_success = False
        state_obj.delta_analyzer_message = f"Delta Analyzer Error: {str(e)}"
        state_obj.delta_analyzer_result = None

    return state_obj.dict()

async def code_generator_node(state: dict) -> dict:
    session_id = state["current_user"]
    trace = get_or_create_trace(session_id)

    state_obj = ensure_state_schema(state)
    logger.info("Code Generator Node: Generating code modifications...")

    try:
        if state_obj.change_type == "code_change":
            if not state_obj.delta_analyzer_result or not state_obj.master_planner_result:
                logger.warning("No Delta Analyzer or Master Planner results available")
                state_obj.code_generator_success = False
                state_obj.code_generator_message = "No Delta Analyzer or Master Planner results available"
                state_obj.code_generator_result = None
                return state_obj.dict()

            # Convert Delta Analyzer output to Code Generator input format
            modification_plan = state_obj.delta_analyzer_result
            generator_input = CodeGeneratorInput(
                modification_plan=modification_plan,
                user_query=state_obj.developer_task
            )

        # Generate code modifications
            result: CodeGeneratorOutput = await code_generator_agent.generate_code_modifications(generator_input)

            generation = trace.generation(
            name="code_generator_agent_step",
            model="claude-3-7-sonnet-latest",
            input=generator_input
            )
            generation.update(output=result.model_dump())
            generation.end()
        else:
            result = {
                "success" : True,
                "modified_files":[
                    {
                        "file_path" : "",
                        "original_content" : "",
                        "modified_content" : state_obj.updated_config,
                        "modifications_applied" :0,
                        "backup_path" : None
                    }
                ],
                "failed_files" : [],
                "errors": [],
                "warnings" : [],
                "total_modifications" : 0,
                "execution_time" : 0
            }
            result = CodeGeneratorOutput(**result)

            generation = trace.generation(
            name="code_generator_agent_step",
            model="claude-3.7-sonnet-latest",
            input=state_obj.updated_config
            )
            generation.update(output=result.model_dump())
            generation.end()


        state_obj.code_generator_result = result
        state_obj.code_generator_success = True
        state_obj.code_generator_message = "Code Generator completed successfully" if result.success else "Code Generator failed"

        logger.info(f"Code Generator Success: {result.success}")
        logger.info(f"Modified Files: {len(result.modified_files)}")
        logger.info(f"Failed Files: {len(result.failed_files)}")

    except Exception as e:
        logger.error(f"Error in Code Generator Node: {e}")
        state_obj.code_generator_success = False
        state_obj.code_generator_message = f"Code Generator Error: {str(e)}"
        state_obj.code_generator_result = None

    return state_obj.dict()

# NEW: Code Validator Node
async def code_validator_node(state: dict) -> dict:
    session_id = state["current_user"]
    trace = get_or_create_trace(session_id)

    state_obj = ensure_state_schema(state)
    logger.info("Code Validator Node: Validating generated code...")

    try:
        if state_obj.change_type == "code_change":
            if not state_obj.code_generator_result or not state_obj.code_generator_success:
                logger.warning("No Code Generator results available for validation")
                state_obj.code_validator_success = False
                state_obj.code_validator_message = "No Code Generator results available for validation"
                state_obj.code_validator_result = None
                return state_obj.dict()

            # Convert Code Generator output to Code Validator input format
            validator_input = convert_generator_to_validator_format(state_obj.code_generator_result)

            # Validate the generated code
            result: CodeValidatorOutput = await code_validator_agent.validate_code_changes(validator_input)

            generation = trace.generation(
            name="code_validator_agent_step",
            model="gpt-4o",
            input=validator_input
            )
            generation.update(output=result.model_dump())
            generation.end()

        else:
            updated_config = state_obj.updated_config
            result = CodeValidatorOutput(success=True,
                                         overall_status="passed",
                                         files_validated=[FileValidationResult(
                                             file_path="",
                                             syntax_valid=True,
                                             errors=[],
                                             warnings=[""],
                                             suggestions=[],
                                             metrics=CodeMetrics(lines_of_code=22,blank_lines=6,comment_lines=2, functions_count=1, classes_count=0, imports_count=4,complexity_estimate='low', complexity_score=0.0),
                                             validation_passed=True
                                         )],
                                         validation_summary=ValidationSummary(total_files=0, files_with_errors=0, files_with_warnings=0,files_passed=0, total_errors=0, total_warnings=0, total_suggestions=0,overall_quality_score=0.0),
                                         errors_found=[],
                                         warnings=[],
                                         suggestions=[],
                                         execution_time= 0.00,
                                         timestamp= datetime.now().isoformat())

            generation = trace.generation(
            name="code_validator_agent_step",
            model="gpt-4o",
            input=updated_config
            )
            generation.update(output=result.model_dump())
            generation.end()

        state_obj.code_validator_result = result
        state_obj.code_validator_success = result.success and (result.overall_status == "passed")
        state_obj.code_validator_message = f"Code Validator completed - Status: {result.overall_status}"

        logger.info(f"Code Validator Success: {state_obj.code_validator_success}")
        logger.info(f"Overall Status: {result.overall_status}")

    except Exception as e:
        logger.error(f"Error in Code Validator Node: {e}")
        state_obj.code_validator_success = False
        state_obj.code_validator_message = f"Code Validator Error: {str(e)}"
        state_obj.code_validator_result = None

    return state_obj.dict()

# ---------- Conditional Logic Functions ----------
def should_proceed_to_master_planner(state: dict) -> str:
    """Determine whether to proceed to master planner or end"""
    state_obj = ensure_state_schema(state)
    if state_obj.is_satisfied:
        return "master_planner"
    else:
        return "__end__"

def should_proceed_to_delta_analyzer(state: Dict[str, Any]) -> str:
    """Ask user confirmation and determine next step"""
    try:
        state_obj = ensure_state_schema(state)

        if not getattr(state_obj, "master_planner_success", False):
            logger.warning("Master planner not successful")
            return "__end__"

        approval = getattr(state_obj, "master_planner_approved", None)

        logger.info(f"Master Planner approval status : {approval}")

        if approval is None:
            logger.info("Master planner is Successful but no approval yet")
            return "__end__"
        if approval is True or (isinstance(approval, str) and approval.lower() in ["yes", "true" , "approve", "y" , "1"]):
            logger.info("Master Planner is approved and proceed to delta analyzer")
            return "delta_analyzer"

        if approval is False or (isinstance(approval, str) and approval.lower() in ["no", "false", "reject" ,"n" , "0"]):
            logger.info("Plan is rejected by the User")

            additional_task = getattr(state_obj, "additional_requirement",None)
            rejection_feedback = getattr(state_obj, "rejection_feedback", None)

            if additional_task or rejection_feedback:
                logger.info("additional feedback is provided")
                return "restart_from_communication"
            else:
                logger.info("No additional feedback, ending workflow")
                return "__end__"

        logger.warning(f"Unexpected approval value: {approval}")
        return "__end__"

    except Exception as e:
        logger.error(f"Erroe in should_proceed_to_delta_analyzer: {e}")
        return "__end__"

def should_proceed_to_code_generator(state: dict) -> str:
    """Handle Delta Analyzer results and proceed to Code Generator if successful"""
    state_obj = ensure_state_schema(state)
    if state_obj.delta_analyzer_success:
        return "code_generator"  # Proceed to Code Generator
    else:
        return "master_planner"  # Delta Analyzer failed - go back to Master Planner

def should_proceed_to_code_validator(state: dict) -> str:
    """Handle Code Generator results and proceed to Code Validator if successful"""
    state_obj = ensure_state_schema(state)
    if state_obj.code_generator_success:
        return "code_validator"  # Proceed to Code Validator
    else:
        return "delta_analyzer"  # Code Generator failed - go back to Delta Analyzer

# UPDATED: Handle Code Validator results with state saving
def should_end_after_validation(state: dict) -> str:
    """Handle Code Validator results - END if passed (and save state), restart from Master Planner if failed"""
    state_obj = ensure_state_schema(state)

    if state_obj.code_validator_success:
        # NEW: Save bot state to ledger when validation passes
        # Get current user from the state or use default
        current_user = getattr(state_obj, 'current_user', 'default_user')

        return "__end__"  # Validation passed - end the flow
    else:
        # Validation failed - show results and get user feedback
        if state_obj.code_validator_result:
            

            # Get user feedback for fixing validation issues
            user_feedback = "continue"

            if user_feedback.lower() in ['skip', 'continue']:
                print("⏭ User chose to skip validation issues. Ending workflow.")
                return "__end__"
            elif user_feedback:
                # Update the developer task with the user feedback
                updated_task = f"{state_obj.developer_task}\n\nAdditional requirements to fix validation issues:\n{user_feedback}"
                state_obj.developer_task = updated_task
                print("🔄 Restarting from Master Planner with updated requirements...")
                return "master_planner"  # Go back to Master Planner with updated requirements
            else:
                print("❌ No feedback provided. Going back to Master Planner with original task.")
                return "master_planner"
        else:
            return "master_planner"  # No validation results - go back to Master Planner


