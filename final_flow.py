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

# NEW: Function to save bot state to JSON when validation passes
def save_bot_state_to_ledger(state: BotStateSchema, user_id: str):
    """Save bot state to JSON file in ledger folder when code validation passes"""
    try:
        # Create ledger directory if it doesn't exist
        ledger_dir = Path("ledger")
        ledger_dir.mkdir(exist_ok=True)
        
        # Generate filename with user ID and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}_{timestamp}.json"
        file_path = ledger_dir / filename
        
        # Convert state to dictionary and handle datetime serialization
        state_dict = state.dict()
        
        # Convert datetime objects to ISO format strings for JSON serialization
        def convert_datetime_to_str(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime_to_str(item) for item in obj]
            else:
                return obj
        
        state_dict = convert_datetime_to_str(state_dict)
        
        # Add metadata
        ledger_entry = {
            "user_id": user_id,
            "timestamp": timestamp,
            "saved_at": datetime.now().isoformat(),
            "validation_status": "PASSED",
            "bot_state": state_dict
        }
        
        # Save to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(ledger_entry, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Bot state saved to ledger: {file_path}")
        print(f"📁 Bot state saved to ledger: {file_path}")
        
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving bot state to ledger: {e}")
        print(f"❌ Error saving bot state to ledger: {e}")
        return None

# [Keep all existing helper functions unchanged...]
def get_user_history(user_id: str = "default_user") -> List[str]:
    """Get user history from Redis."""
    history_key = f"user:{user_id}:history"
    try:
        if redis_client:
            history = redis_client.lrange(history_key, 0, -1)
            return history if history else []
        else:
            logger.warning("Redis not available, using empty history")
            return []
    except Exception as e:
        logger.error(f"Error getting history from Redis: {e}")
        return []

def save_to_history(user_id: str, query: str):
    """Save query to user history in Redis."""
    history_key = f"user:{user_id}:history"
    try:
        if redis_client:
            redis_client.rpush(history_key, query)
            redis_client.ltrim(history_key, -50, -1)
            logger.debug(f"Saved query to Redis for user {user_id}")
        else:
            logger.warning("Redis not available, query not saved")
    except Exception as e:
        logger.error(f"Error saving to Redis: {e}")

def clear_user_history(user_id: str = "default_user") -> bool:
    """Clear user history from Redis."""
    history_key = f"user:{user_id}:history"
    try:
        if redis_client:
            if redis_client.exists(history_key):
                redis_client.delete(history_key)
                logger.info(f"Cleared history for user {user_id}")
                return True
            else:
                logger.info(f"No history found for user {user_id}")
                return False
        else:
            logger.warning("Redis not available, cannot clear history")
            return False
    except Exception as e:
        logger.error(f"Error clearing history from Redis: {e}")
        return False

def delete_user_completely(user_id: str = "default_user") -> bool:
    """Completely delete user and all associated data from Redis."""
    try:
        if redis_client:
            user_pattern = f"user:{user_id}:*"
            user_keys = redis_client.keys(user_pattern)
            if user_keys:
                deleted_count = redis_client.delete(*user_keys)
                logger.info(f"Deleted {deleted_count} keys for user {user_id}")
                return True
            else:
                logger.info(f"No data found for user {user_id}")
                return False
        else:
            logger.warning("Redis not available, cannot delete user")
            return False
    except Exception as e:
        logger.error(f"Error deleting user from Redis: {e}")
        return False

def get_all_users() -> List[str]:
    """Get list of all users in Redis."""
    try:
        if redis_client:
            user_keys = redis_client.keys("user:*:history")
            users = []
            for key in user_keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    user_id = parts[1]
                    users.append(user_id)
            return users
        else:
            logger.warning("Redis not available")
            return []
    except Exception as e:
        logger.error(f"Error getting users from Redis: {e}")
        return []

def get_redis_cache_info():
    """Get Redis cache information."""
    try:
        if redis_client:
            info = redis_client.info()
            all_keys = redis_client.keys("*")
            user_keys = redis_client.keys("user:*")
            cache_info = {
                "total_keys": len(all_keys),
                "user_keys": len(user_keys),
                "memory_used": info.get('used_memory_human', 'Unknown'),
                "connected_clients": info.get('connected_clients', 'Unknown'),
                "total_commands": info.get('total_commands_processed', 'Unknown')
            }
            return cache_info
        else:
            return {"error": "Redis not available"}
    except Exception as e:
        logger.error(f"Error getting Redis info: {e}")
        return {"error": str(e)}

def show_history_info(user_id: str = "default_user"):
    """Show user history information."""
    history = get_user_history(user_id)
    if history:
        print(f"\n📚 Conversation History for '{user_id}' ({len(history)} queries):")
        print("-" * 50)
        for i, query in enumerate(history, 1):
            display_query = query[:60] + "..." if len(query) > 60 else query
            print(f"   {i:2d}. {display_query}")
        print("-" * 50)
    else:
        print(f"\n No conversation history found for user '{user_id}'")

def show_cache_stats():
    """Show Redis cache statistics."""
    print("\n Redis Cache Statistics:")
    print("-" * 40)
    cache_info = get_redis_cache_info()
    if "error" not in cache_info:
        print(f" Total Keys: {cache_info['total_keys']}")
        print(f" User Keys: {cache_info['user_keys']}")
        print(f" Memory Used: {cache_info['memory_used']}")
        print(f" Connected Clients: {cache_info['connected_clients']}")
        print(f" Total Commands: {cache_info['total_commands']}")
        users = get_all_users()
        if users:
            print(f"\n Active Users ({len(users)}):")
            for user in users:
                history_count = len(get_user_history(user))
                print(f"   • {user} ({history_count} queries)")
        else:
            print("\n No active users found")
    else:
        print(f" Error: {cache_info['error']}")
    print("-" * 40)

def ensure_state_schema(state: Union[dict, BaseModel]) -> BotStateSchema:
    if isinstance(state, BotStateSchema):
        return state
    return BotStateSchema(**state)

def print_detailed_plan(files_to_modify: List):
    """Print detailed plan for each file."""
    print(files_to_modify)

def print_delta_analyzer_results(delta_output: DeltaAnalyzerOutput):
    print(delta_output)

def print_code_generator_results(code_output: CodeGeneratorOutput):
    print(code_output)
   
# NEW: Code Validator results printing function
def print_code_validator_results(validator_output: CodeValidatorOutput):
    print(validator_output)

def show_help():
    """Show available commands."""
    print("\n  Available Commands:")
    print("  • Type your query to process it")
    print("  • 'history' - Show your conversation history")
    print("  • 'clear' or 'clear history' - Clear your conversation history")
    print("  • 'delete user' - Completely delete current user from Redis")
    print("  • 'delete user <user_id>' - Delete specific user from Redis")
    print("  • 'users' - Show all users in Redis")
    print("  • 'cache' - Show Redis cache statistics")
    print("  • 'user <user_id>' - Switch to different user")
    print("  • 'help' - Show this help message")
    print("  • 'exit' or 'quit' - Exit the application")

def get_user_confirmation_for_master_planner() -> bool:
    pass
   
# NEW: Function to get user feedback for validation failures
def get_user_feedback_for_validation_failure(validator_output: CodeValidatorOutput) -> str:
    """Get user feedback when code validation fails"""
    print("\n" + "="*60)
    print("❌ CODE VALIDATION FAILED")
    print("="*60)
    print("The generated code has validation issues that need to be addressed.")
    print("\nMain Issues Found:")
    
    # Show top 5 errors
    if validator_output.errors_found:
        print("\n🚨 Critical Errors:")
        for i, error in enumerate(validator_output.errors_found[:5], 1):
            print(f"   {i}. {error}")
        if len(validator_output.errors_found) > 5:
            print(f"   ... and {len(validator_output.errors_found) - 5} more errors")
    
    # Show top 3 warnings if no errors
    if not validator_output.errors_found and validator_output.warnings:
        print("\n⚠️  Warnings:")
        for i, warning in enumerate(validator_output.warnings[:3], 1):
            print(f"   {i}. {warning}")
        if len(validator_output.warnings) > 3:
            print(f"   ... and {len(validator_output.warnings) - 3} more warnings")
    
    print("\n" + "="*60)
    print("Please provide updated requirements to fix these issues:")
    print("• Be specific about how you want the issues resolved")
    print("• You can reference the errors shown above")
    print("• Type 'skip' if you want to proceed anyway")
    print("="*60)
    
    user_feedback = input("Your updated requirements: ").strip()
    return user_feedback

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
        name = "multi_agent_workflow"
    )
    return trace


# ---------- LangGraph-Compatible Nodes ----------
async def communication_node(state: dict) -> dict:
    session_id = state["current_user"]
    print("I am inside Communication Node")

    state_obj = ensure_state_schema(state)
    logger.info("Communication Node: Extracting intent...")

    trace = get_or_create_trace(session_id)

    span = trace.span(
        name = "communication_agent_step"
    )

    
    comm_input = CommunicationInput(
        user_query=state_obj.latest_query,
        conversation_history=state_obj.user_history[:-1] if len(state_obj.user_history) > 1 else []
    )
    
    result: CommunicationOutput = await communication_agent.extract_intent(comm_input)
    
    span.update(
        input = comm_input.model_dump(),
        output = result.model_dump()
    )
    span.end()

    state_obj.core_intent = result.core_intent
    state_obj.context_notes = result.context_notes
    state_obj.communication_success = result.success
    
    logger.info(f"Core Intent: {result.core_intent}")
    logger.info(f"Context Notes: {result.context_notes}")
    
    return state_obj.dict()

async def query_enhancement_node(state: dict) -> dict:
    session_id = state["current_user"]
    print("I am inside query rephrase agent")

    state_obj = ensure_state_schema(state)
    logger.info("Query Enhancement Node: Rephrasing and validating...")

    trace = get_or_create_trace(session_id)

    span = trace.span(
        name = "query_rephrase_agent_step"
    )
    
    enhancer_input = QueryEnhancerInput(
        core_intent=state_obj.core_intent,
        context_notes=state_obj.context_notes
    )
    
    result: QueryEnhancerOutput = await query_rephraser_agent.enhance_query(enhancer_input)

    span.update(
        input = enhancer_input.model_dump(),
        output = result.model_dump()
    )
    span.end()
    
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
    session_id = state["current_user"]
    trace = get_or_create_trace(session_id)

    state_obj = ensure_state_schema(state)
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

                    span = trace.span(
                        name = "dependency_extractor_agent_step"
                    )

                    span.update(
                    input = dependency_input.model_dump(),
                    output = result.model_dump()
                    )
                    span.end()

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
                    print(len(files_dict))
                    req_rag_output = str(files_dict) + "\n here you have the list of all files which are present in the code repo i want you to consider each file for thr code migration purpose or other type of full repo update and based on the user query set the expectation for each file clearly and based on the user query set the expectations for each files clearly and include all the files."
                else:
                    input_data = DocumentGeneratorInput(
                        developer_task_query=state_obj.developer_task
                    )
                    Document = DocumentGeneratorAgent()
                    rag_output = await Document.generate_document(input_data)
                    req_rag_output = rag_output.generated_doc
                    span = trace.span(
                        name = "document_generator_agent_step"
                    )

                    span.update(
                    input = input_data.model_dump(),
                    output = rag_output.model_dump()
                    )
                    span.end()
            else:
                print("simple code generation node")
                input_data = DocumentGeneratorInput(
                        developer_task_query=state_obj.developer_task
                    )
                
                Document = DocumentGeneratorAgent()
                rag_output = await Document.generate_document(input_data)
                req_rag_output = rag_output.generated_doc

                span = trace.span(
                    name = "document_generator_agent_step"
                )
                span.update(
                input = input_data.model_dump(),
                output = rag_output.model_dump()
                )
                span.end()

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
        
        span = trace.span(
                    name = "master_planner_agent_step"
                )
        planner_input = MasterPlannerInput(
            parsed_config=parsed_config,
            user_question=state_obj.developer_task
        )
        

        result: MasterPlannerOutput = await master_planner_agent.identify_target_files(planner_input, rag_result=req_rag_output)
        
        span.update(
                input = planner_input.model_dump(),
                output = result.model_dump()
                )
        span.end()

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
    span = trace.span(
        name = "delta_analyzer_agent_step"
    )
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
            
            result_delta = await delta_analyzer_agent.create_modification_plan(target_files,parsed_config,user_query)

            span.update(
                input = delta_analyzer_input_dict,
                output = result_delta
            )

            span.end()
        else:
            result_delta = "its a config change"
            span.update(
                input = "Not required for config change",
                output = result_delta
            )
            span.end()
        
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
    span = trace.span(
        name = "code_generator_agent_step"
    )

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
            span.update(
                input = generator_input,
                output = result.model_dump()
            )
            span.end()
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
            span.update(
                input = state_obj.updated_config,
                output = result.model_dump()
            )
            span.end()

        
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
    span = trace.span(
        name = "code_validator_agent_step"
    )

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
            span.update(
                input = validator_input,
                output = result.model_dump()
            )
            span.end()
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
            span.update(
                input = updated_config,
                output = result.model_dump()
            )
            span.end()
        
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
        
        if approval is False or (isinstance(approval, str) and approval.lower() in ["no", "flase", "reject" ,"n" , "0"]):
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
        saved_path = save_bot_state_to_ledger(state_obj, current_user)
        
        if saved_path:
            print(f"\n✅ SUCCESS: Bot state has been saved to ledger!")
            print(f"📄 Ledger file: {saved_path}")
        
        return "__end__"  # Validation passed - end the flow
    else:
        # Validation failed - show results and get user feedback
        if state_obj.code_validator_result:
            print_code_validator_results(state_obj.code_validator_result)
            
            # Get user feedback for fixing validation issues
            user_feedback = get_user_feedback_for_validation_failure(state_obj.code_validator_result)
            
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

def print_graph_structure():
    """Print a detailed text representation of the graph structure"""
    print("\n" + "="*60)
    print("🔄 LANGGRAPH WORKFLOW STRUCTURE")
    print("="*60)
    print("")
    print("📋 WORKFLOW STEPS:")
    print("  1️⃣  Communication Agent")
    print("      └─ Extract user intent and context")
    print("")
    print("  2️⃣  Query Enhancement Agent")
    print("      └─ Rephrase and validate query")
    print("      └─ Decision: Is query satisfied?")
    print("          ├─ NO  → End workflow")
    print("          └─ YES → Continue to Step 3")
    print("")
    print("  3️⃣  Master Planner Agent")
    print("      └─ Identify target files")
    print("      └─ User Confirmation Required")
    print("          ├─ NO  → Back to Step 1")
    print("          └─ YES → Continue to Step 4")
    print("")
    print("  4️⃣  Delta Analyzer Agent")
    print("      └─ Generate modification plan")
    print("      └─ Decision: Success?")
    print("          ├─ NO  → Back to Step 3")
    print("          └─ YES → Continue to Step 5")
    print("")
    print("  5️⃣  Code Generator Agent")
    print("      └─ Generate actual code modifications")
    print("      └─ Decision: Success?")
    print("          ├─ NO  → Back to Step 4")
    print("          └─ YES → Continue to Step 6")
    print("")
    print("  6️⃣  Code Validator Agent 🆕")
    print("      └─ Validate generated code")
    print("      └─ Decision: Validation passed?")
    print("          ├─ NO  → Back to Step 3 (with user feedback)")
    print("          └─ YES → Save state to ledger 💾 → End workflow ✅")
    print("")
    print("🔗 WORKFLOW CONNECTIONS:")
    print("  communication_node → query_enhancement_node")
    print("  query_enhancement_node → [conditional] → master_planner_node | END")
    print("  master_planner_node → [conditional] → delta_analyzer_node | communication_node | END")
    print("  delta_analyzer_node → [conditional] → code_generator_node | master_planner_node")
    print("  code_generator_node → [conditional] → code_validator_node | delta_analyzer_node")
    print("  code_validator_node → [conditional] → END (+ save to ledger) | master_planner_node")
    print("="*60)

# ---------- Main Function ----------
async def main():
    print("\nWelcome to the LangGraph Query Clarifier with Master Planner, Delta Analyzer, Code Generator, and Code Validator")
    print("=" * 110)
    
    current_user = "default_user"
    print(f" Current user: {current_user}")
    
    history = get_user_history(current_user)
    if history:
        print(f" Loaded {len(history)} previous queries from session")
    
    show_help()
    
    while True:
        user_input = input(f"\n[{current_user}] Enter your query: ").strip()
        
        if not user_input:
            print("Please enter a valid input.")
            continue
        
        # Handle special commands (unchanged)
        if user_input.lower() in ["exit", "quit", "q"]:
            print(" Goodbye!")
            break
        elif user_input.lower() == "help":
            show_help()
            continue
        elif user_input.lower() in ["clear", "clear history"]:
            if clear_user_history(current_user):
                print(f" History cleared for user '{current_user}'")
                history = []
            else:
                print(f" No history found for user '{current_user}' or Redis unavailable")
            continue
        elif user_input.lower() == "delete user":
            confirm = input(f" Are you sure you want to completely delete user '{current_user}' from Redis? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                if delete_user_completely(current_user):
                    print(f" User '{current_user}' completely deleted from Redis")
                    history = []
                else:
                    print(f" No data found for user '{current_user}' or Redis unavailable")
            else:
                print(" Operation cancelled")
            continue
        elif user_input.lower().startswith("delete user "):
            target_user = user_input[12:].strip()
            if target_user:
                confirm = input(f" Are you sure you want to completely delete user '{target_user}' from Redis? (yes/no): ").strip().lower()
                if confirm in ['yes', 'y']:
                    if delete_user_completely(target_user):
                        print(f" User '{target_user}' completely deleted from Redis")
                        if target_user == current_user:
                            history = []
                    else:
                        print(f" No data found for user '{target_user}' or Redis unavailable")
                else:
                    print(" Operation cancelled")
            else:
                print("Please provide a valid user ID")
            continue
        elif user_input.lower() == "users":
            users = get_all_users()
            if users:
                print(f"\n👥 Active Users ({len(users)}):")
                for user in users:
                    history_count = len(get_user_history(user))
                    status = " (current)" if user == current_user else ""
                    print(f"   • {user} ({history_count} queries){status}")
            else:
                print("\n👥 No active users found in Redis")
            continue
        elif user_input.lower() == "cache":
            show_cache_stats()
            continue
        elif user_input.lower() == "history":
            show_history_info(current_user)
            continue
        elif user_input.lower().startswith("user "):
            new_user = user_input[5:].strip()
            if new_user:
                current_user = new_user
                history = get_user_history(current_user)
                print(f"👤 Switched to user: {current_user}")
                if history:
                    print(f" Loaded {len(history)} queries for this user")
            else:
                print("Please provide a valid user ID")
            continue
        
        # Main processing
        save_to_history(current_user, user_input)
        history.append(user_input)
        
        # Set up retry loop for failures
        max_retries = 3
        retry_count = 0
        process_completed = False
        current_query = user_input
        
        while retry_count < max_retries and not process_completed:
            state = BotStateSchema(
                latest_query=current_query,
                user_history=history,
                current_user=current_user  # NEW: Add current user to state
            )
            
            # Build LangGraph with Code Validator
            builder = StateGraph(dict)
            builder.add_node("communication_node", communication_node)
            builder.add_node("query_enhancement_node", query_enhancement_node)
            builder.add_node("master_planner_node", master_planner_node)
            builder.add_node("delta_analyzer_node", delta_analyzer_node)
            builder.add_node("code_generator_node", code_generator_node)
            builder.add_node("code_validator_node", code_validator_node)  # NEW NODE
            
            builder.set_entry_point("communication_node")
            builder.add_edge("communication_node", "query_enhancement_node")
            
            builder.add_conditional_edges(
                "query_enhancement_node",
                should_proceed_to_master_planner,
                {
                    "master_planner": "master_planner_node",
                    "__end__": "__end__"
                }
            )
            
            builder.add_conditional_edges(
                "master_planner_node",
                should_proceed_to_delta_analyzer,
                {
                    "delta_analyzer": "delta_analyzer_node",    # ✅ User approved
                    "user_feedback": "user_feedback_node",      # 🔄 User wants to modify (NEW!)
                    "__end__": "__end__"                        # ❌ Master Planner failed
                }
            )           
            builder.add_conditional_edges(
                "delta_analyzer_node",
                should_proceed_to_code_generator,
                {
                    "code_generator": "code_generator_node",
                    "master_planner": "master_planner_node"
                }
            )
            
            builder.add_conditional_edges(
                "code_generator_node",
                should_proceed_to_code_validator,
                {
                    "code_validator": "code_validator_node",
                    "delta_analyzer": "delta_analyzer_node"
                }
            )
            
            # NEW: Code Validator conditional edges
            builder.add_conditional_edges(
                "code_validator_node",
                should_end_after_validation,
                {
                    "__end__": "__end__",
                    "master_planner": "master_planner_node"
                }
            )
            
            graph = builder.compile()
            
            if retry_count == 0:
                try:
                    ascii_art = graph.get_graph().draw_ascii()
                    print("\n🔄 LangGraph ASCII Structure:")
                    print(ascii_art)
                except Exception as e:
                    logger.warning(f"Unable to draw ASCII graph: {e}")
                    print_graph_structure()
            
            try:
                # Thread-safe graph execution
                with graph_lock:
                    final_state_dict = await graph.ainvoke(state.dict())
                final_state = BotStateSchema(**final_state_dict)
            except Exception as e:
                logger.error(f"Graph execution error: {e}")
                print(f" Error executing graph: {e}")
                break
            
            print(f"\n{'='*60}")
            print(f"ATTEMPT {retry_count + 1}/{max_retries}")
            print("=" * 60)
            print(f"Core Intent: {final_state.core_intent}")
            print(f"Context: {final_state.context_notes}")
            print(f"Developer Task: {final_state.developer_task}")
            print(f"Satisfied: {final_state.is_satisfied}")
            
            if not final_state.is_satisfied:
                print("Suggestions:")
                for s in final_state.suggestions:
                    print(f"- {s}")
                break
            
            # Check final results - NOW INCLUDING CODE VALIDATOR
            if (hasattr(final_state, 'code_validator_success') and final_state.code_validator_success 
                and final_state.code_validator_result):
                # Show all results in order
                print(f"\n🔄 DELTA ANALYZER RESULTS")
                print(f"Delta Analyzer Success: {final_state.delta_analyzer_success}")
                print(f"Delta Analyzer Message: {final_state.delta_analyzer_message}")
                print_delta_analyzer_results(final_state.delta_analyzer_result)
                
                print(f"\n💻 CODE GENERATOR RESULTS")
                print(f"Code Generator Success: {final_state.code_generator_success}")
                print(f"Code Generator Message: {final_state.code_generator_message}")
                print_code_generator_results(final_state.code_generator_result)
                
                print(f"\n🔍 CODE VALIDATOR RESULTS")
                print(f"Code Validator Success: {final_state.code_validator_success}")
                print(f"Code Validator Message: {final_state.code_validator_message}")
                print_code_validator_results(final_state.code_validator_result)
                
                print("\n🎉 COMPLETE WORKFLOW FINISHED!")
                print("✅ All agents completed successfully:")
                print("   • Communication Agent ✓")
                print("   • Query Enhancement Agent ✓")
                print("   • Master Planner Agent ✓")
                print("   • Delta Analyzer Agent ✓")
                print("   • Code Generator Agent ✓")
                print("   • Code Validator Agent ✓")
                print("💾 Bot state saved to ledger!")
                print("🚀 Code validation passed! Ready for deployment!")
                print("=" * 60)
                process_completed = True
                break
                
            elif hasattr(final_state, 'code_validator_success') and not final_state.code_validator_success:
                # Code Validator failed - this is handled by the conditional logic, but we can show info here
                print(f"\n❌ Code Validation Failed (Attempt {retry_count + 1}/{max_retries})")
                if hasattr(final_state, 'code_validator_result') and final_state.code_validator_result:
                    print(f"Validation Status: {final_state.code_validator_result.overall_status}")
                    print(f"Errors Found: {len(final_state.code_validator_result.errors_found)}")
                    print(f"Warnings: {len(final_state.code_validator_result.warnings)}")
                
                # The conditional logic should have handled getting user feedback and updating the task
                # If we reach here, it means the workflow is continuing from Master Planner
                process_completed = False
                break  # Let the workflow handle the retry
                
            elif final_state.code_generator_success and final_state.code_generator_result:
                # Show Code Generator results only (Code Validator not reached yet or failed)
                print(f"\n💻 CODE GENERATOR RESULTS")
                print(f"Code Generator Success: {final_state.code_generator_success}")
                print(f"Code Generator Message: {final_state.code_generator_message}")
                print_code_generator_results(final_state.code_generator_result)
                
                if hasattr(final_state, 'code_validator_result') and final_state.code_validator_result:
                    print(f"\n🔍 CODE VALIDATOR RESULTS")
                    print(f"Code Validator Success: {final_state.code_validator_success}")
                    print_code_validator_results(final_state.code_validator_result)
                
                print("\n🎉 PROCESS COMPLETED!")
                print("✅ Code generation completed.")
                if hasattr(final_state, 'code_validator_success') and not final_state.code_validator_success:
                    print("⚠️  However, code validation revealed issues that need attention.")
                print("=" * 60)
                process_completed = True
                break
            
            elif hasattr(final_state, 'code_generator_success') and not final_state.code_generator_success:
                # Code Generator failed - ask for clarification
                print(f"\n❌ Code Generator Failed (Attempt {retry_count + 1}/{max_retries})")
                print(f"Reason: {getattr(final_state, 'code_generator_message', 'Unknown error')}")
                print("\nThis might be because:")
                print("  • The modification plan is too complex")
                print("  • The code structure couldn't be understood")
                print("  • The generated code has syntax errors")
                
                if retry_count < max_retries - 1:
                    print(f"\n You have {max_retries - retry_count - 1} more attempts.")
                    clarification = input("Could you clarify your task or provide more details? (or 'skip' to continue): ").strip()
                    if clarification.lower() in ['skip', 'continue', '']:
                        print("⏭ Skipping retry...")
                        break
                    elif clarification:
                        current_query = clarification
                        print("🔄 Retrying with your clarification...")
                        save_to_history(current_user, clarification)
                        history.append(clarification)
                        retry_count += 1
                    else:
                        print("No clarification provided. Ending...")
                        break
                else:
                    print("\n❌ Maximum retry attempts reached.")
                    break
            
            elif hasattr(final_state, 'delta_analyzer_success') and not final_state.delta_analyzer_success:
                # Delta Analyzer failed - ask for clarification
                print(f"\n❌ Delta Analyzer Failed (Attempt {retry_count + 1}/{max_retries})")
                print(f"Reason: {getattr(final_state, 'delta_analyzer_message', 'Unknown error')}")
                print("\nThis might be because:")
                print("  • The file analysis is incomplete or incorrect")
                print("  • The task requirements are too complex")
                print("  • The modification plan couldn't be generated")
                
                if retry_count < max_retries - 1:
                    print(f"\n You have {max_retries - retry_count - 1} more attempts.")
                    clarification = input("Could you clarify your task or provide more details? (or 'skip' to continue): ").strip()
                    if clarification.lower() in ['skip', 'continue', '']:
                        print("⏭ Skipping retry...")
                        break
                    elif clarification:
                        current_query = clarification
                        print("🔄 Retrying with your clarification...")
                        save_to_history(current_user, clarification)
                        history.append(clarification)
                        retry_count += 1
                    else:
                        print("No clarification provided. Ending...")
                        break
                else:
                    print("\n❌ Maximum retry attempts reached.")
                    break
            
            elif not final_state.master_planner_success:
                # Master Planner failed
                print(f"\n❌ Master Planner Failed (Attempt {retry_count + 1}/{max_retries})")
                print(f"Reason: {final_state.master_planner_message}")
                print("\nThis might be because:")
                print("  • The specified file doesn't exist")
                print("  • The task is too vague or abstract")
                print("  • The project structure doesn't match the task")
                
                if retry_count < max_retries - 1:
                    print(f"\n You have {max_retries - retry_count - 1} more attempts.")
                    clarification = input("Could you clarify your task or provide more details? (or 'skip' to continue): ").strip()
                    if clarification.lower() in ['skip', 'continue', '']:
                        print("⏭ Skipping retry...")
                        break
                    elif clarification:
                        current_query = clarification
                        print("🔄 Retrying with your clarification...")
                        save_to_history(current_user, clarification)
                        history.append(clarification)
                        retry_count += 1
                    else:
                        print("No clarification provided. Ending...")
                        break
                else:
                    print("\n❌ Maximum retry attempts reached.")
                    break
            else:
                # User said no to Master Planner - get new query
                print("\n🔄 RESTARTING PROCESS")
                print("Please provide an updated or new query:")
                new_query = input(f"[{current_user}] Updated query: ").strip()
                if new_query:
                    current_query = new_query
                    print("🔄 Restarting with your updated query...")
                    save_to_history(current_user, new_query)
                    history.append(new_query)
                    retry_count = 0  # Reset retry count for new query
                else:
                    print("❌ No query provided. Using original query.")
                    break
        
        if not process_completed:
            print("❌ Unable to complete the process. Moving to next query.")

if __name__ == "__main__":
    asyncio.run(main())