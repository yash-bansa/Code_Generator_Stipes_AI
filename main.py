from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
import uvicorn 
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import asyncio
from enum import Enum
import json
import logging
from datetime import datetime
from pathlib import Path
from utils.llm_client import llm_client
import ast
from fastapi.responses import StreamingResponse
from threading import Lock
# Import your existing modules
from agents import *
from config.agents_io import *
from config.settings import settings
app = FastAPI(title="LangGraph Interactive Code Assistant API", version="1.0.0")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("langraph_api.log")
    ]
)
logger = logging.getLogger(__name__)

graph_lock = Lock()

# Data Models for API
class QueryType(str, Enum):
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    FEEDBACK = "feedback"

class UserQuery(BaseModel):
    session_id: str
    message: str
    load_history_from: Optional[str] = None  # User ID to load history from

def get_additional_requirements(text):
    stripped_text = text.lstrip()

    patterns = ["no", "no," , "no."]

    for pattern in patterns:
        if stripped_text.lower().startswith(pattern):
            return stripped_text[len(pattern):].lstrip()

    return ""    

class APIResponse(BaseModel):
    session_id: str
    status: str  # "processing", "waiting_for_input", "completed", "error", "query_refinement_needed", etc.
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    required_input: Optional[Dict[str, Any]] = None
    next_step: Optional[str] = None
    workflow_stage: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class SessionInfo(BaseModel):
    session_id: str
    status: str
    last_activity: datetime
    current_stage: str
    query_count: int

# Initialize agents
logger.info("Initializing agents...")
communication_agent = CommunicationAgent()
query_rephraser_agent = QueryRephraserAgent()
master_planner_agent = MasterPlannerAgent()
delta_analyzer_agent = DeltaAnalyzerAgent()
code_generator_agent = CodeGeneratorAgent()
code_validator_agent = CodeValidatorAgent()
redis_client = settings.get_redis_connection()
logger.info("Agents initialized successfully!")
# Import session manager and workflow handler
from session_manager_redis import session_manager_redis
from workflow_handler import workflow_handler
# Import your existing helper functions
from final_flow import (
    get_user_history, save_to_history, save_bot_state_to_ledger
)

async def _generate_response(session_id: str, state: BotStateSchema, session: Dict) -> APIResponse:
    """Generate appropriate API response based on workflow state"""
    # PRIORITY CHECK: Master planner approval pending (check this FIRST)
    if (hasattr(state, 'master_planner_success') and state.master_planner_success and
        (not hasattr(state, 'master_planner_approved') or state.master_planner_approved is None)):
        logger.info("Master planner successful, waiting for user approval")
        # Format master planner results for display
        files_info = []
        if hasattr(state, 'master_planner_result') and state.master_planner_result:
            try:
                for file_info in state.master_planner_result:
                    files_info.append({
                        "file_path": file_info.file_path,
                        "priority": file_info.priority,
                        "analysis": file_info.analysis.dict() if hasattr(file_info, 'analysis') else None,
                        "description": getattr(file_info, 'description', 'No description available'),
                        "estimated_changes": getattr(file_info, 'change_summary', 'Unknown changes'),
                        "risk_level": getattr(file_info, 'risk_assessment', 'Medium')
                    })
            except Exception as e:
                logger.error(f"Error formatting master planner results: {e}")
                files_info = [{"error": "Could not format results"}]
        # Update session AFTER preparing response data
        session_manager_redis.update_session(session_id, {
            "waiting_for_input": True,
            "current_stage": "master_planner_approval"
        })
        return APIResponse(
            session_id=session_id,
            status="waiting_for_input",
            message="Master Planner has analyzed your request and identified files to modify. Please review the proposed changes below and decide whether to proceed.",
            workflow_stage="master_planner_approval",
            required_input={
                "type": "master_planner_approval",
                "prompt": "Do you want to proceed with this Master Planner analysis?",
                "options": [
                    {"value": "yes", "label": "Yes, proceed with these changes"},
                    {"value": "no", "label": "No, I want to add more requirements or modify the approach"}
                ],
                "fields": ["master_planner_approved"],
                "rejection_fields": ["rejection_reason", "additional_requirements"],
                "field_descriptions": {
                    "rejection_reason": "Why are you rejecting this approach?",
                    "additional_requirements": "What additional requirements or changes do you want?"
                }
            },
            data={
                "files_to_modify": files_info,
                "total_files": len(files_info),
                "master_planner_summary": {
                    "approach": getattr(state, 'approach_summary', 'Standard file modification approach'),
                    "estimated_complexity": "Medium",
                    "dependencies_affected": []
                },
                "feedback_instructions": {
                    "rejection_guide": "If you reject, please provide specific feedback about what you want changed or added",
                    "examples": [
                        "Add user role management",
                        "Use different authentication method",
                        "Include additional security features",
                        "Modify the database schema approach"
                    ]
                }
            }
        )
    # Check for rejection restart scenario
    if (hasattr(state, 'master_planner_approved') and state.master_planner_approved == False):
        logger.info("Master planner rejected, workflow is restarting with feedback")
        return APIResponse(
            session_id=session_id,
            status="restarting",
            message="Master planner proposal rejected. Restarting workflow with your feedback and additional requirements...",
            workflow_stage="restarting_after_rejection",
            data={
                "reason": "master_planner_rejected",
                "preserved_history": len(state.user_history) if hasattr(state, 'user_history') else 0,
                "enhanced_task": getattr(state, 'developer_task', ''),
                "feedback_incorporated": True
            }
        )
    # Check if workflow ended due to query enhancement failure
    if (hasattr(state, 'is_satisfied') and not state.is_satisfied and
        hasattr(state, 'enhancement_success') and state.enhancement_success and
        hasattr(state, 'suggestions') and state.suggestions):
        session_manager_redis.update_session(session_id, {
            "waiting_for_input": True,
            "current_stage": "query_refinement_needed"
        })
        return APIResponse(
            session_id=session_id,
            status="query_refinement_needed",
            message="Your query needs more clarity to proceed. Please review the suggestions and provide a more detailed request.",
            workflow_stage="query_refinement",
            required_input={
                "type": "query_refinement",
                "prompt": "Please provide a more specific query based on the suggestions below:",
                "fields": ["refined_query"],
                "options": ["retry", "cancel"]
            },
            data={
                "original_query": state.latest_query,
                "core_intent": getattr(state, 'core_intent', 'Unknown'),
                "context_notes": getattr(state, 'context_notes', 'No context available'),
                "suggestions": state.suggestions,
                "change_type" : getattr(state, "change_type", "Invalid"),
                "analysis": {
                    "communication_success": getattr(state, 'communication_success', False),
                    "enhancement_success": getattr(state, 'enhancement_success', False),
                    "is_satisfied": state.is_satisfied,
                    "reason": "Query was processed but lacks sufficient detail for code generation"
                }
            }
        )
    # Check if communication agent failed completely
    if (hasattr(state, 'communication_success') and not state.communication_success):
        return APIResponse(
            session_id=session_id,
            status="communication_failed",
            message="Unable to understand your request. Please try rephrasing with more context.",
            workflow_stage="communication_failed",
            data={
                "original_query": getattr(state, 'latest_query', 'Unknown'),
                "error": "Communication agent could not extract meaningful intent",
                "suggestions": [
                    "Be more specific about what you want to build",
                    "Mention the technology/framework you're using",
                    "Describe the specific functionality you need",
                    "Include context about your current codebase"
                ]
            }
        )
    # Check if query enhancement agent failed
    if (hasattr(state, 'enhancement_success') and not state.enhancement_success):
        return APIResponse(
            session_id=session_id,
            status="enhancement_failed",
            message="Query enhancement failed. Please provide a clearer request.",
            workflow_stage="enhancement_failed",
            data={
                "original_query": getattr(state, 'latest_query', 'Unknown'),
                "core_intent": getattr(state, 'core_intent', 'Unknown'),
                "context_notes": getattr(state, 'context_notes', 'No context available'),
                "error": "Query enhancement agent encountered an error",
                "change_type" : getattr(state, "change_type", "Invalid"),
                "suggestions": [
                    "Try breaking down your request into smaller, specific tasks",
                    "Provide more technical details about your requirements",
                    "Specify the files or components you want to modify"
                ]
            }
        )
    # Check if code validation failed and needs user feedback
    if (hasattr(state, 'code_validator_success') and not state.code_validator_success and
        hasattr(state, 'code_validator_result') and state.code_validator_result is not None):
        validation_result = state.code_validator_result
        # Safety check for validation_summary
        if not hasattr(validation_result, 'validation_summary') or validation_result.validation_summary is None:
            logger.error("Code validator result missing validation_summary")
            return APIResponse(
                session_id=session_id,
                status="error",
                message="Code validation failed due to missing validation summary. Please try again.",
                workflow_stage="validation_error",
                data={
                    "error": "validation_summary_missing",
                    "suggestion": "Restart the workflow or contact support"
                }
            )
        session_manager_redis.update_session(session_id, {
            "waiting_for_input": True,
            "current_stage": "validation_feedback",
            "validation_attempts": session.get("validation_attempts", 0) + 1
        })
        return APIResponse(
            session_id=session_id,
            status="validation_failed",
            message="Code validation failed. Please provide feedback to fix the issues.",
            workflow_stage="validation_feedback",
            required_input={
                "type": "validation_feedback",
                "prompt": "Please provide updated requirements to fix these validation issues:",
                "fields": ["validation_feedback"]
            },
            data={
                "validation_summary": validation_result.validation_summary.dict(),
                "errors": validation_result.errors_found[:5] if hasattr(validation_result, 'errors_found') else [],
                "warnings": validation_result.warnings[:3] if hasattr(validation_result, 'warnings') else [],
                "overall_status": getattr(validation_result, 'overall_status', 'unknown'),
                "quality_score": validation_result.validation_summary.overall_quality_score if hasattr(validation_result.validation_summary, 'overall_quality_score') else 0
            }
        )
    # Check if workflow completed successfully
    if (hasattr(state, 'code_validator_success') and state.code_validator_success and
        hasattr(state, 'code_validator_result') and state.code_validator_result is not None):
        # Safety check for validation_summary
        if not hasattr(state.code_validator_result, 'validation_summary') or state.code_validator_result.validation_summary is None:
            logger.error("Successful validation missing validation_summary")
            return APIResponse(
                session_id=session_id,
                status="error",
                message="Validation completed but summary is missing. Results may be incomplete.",
                workflow_stage="validation_summary_error",
                data={
                    "error": "validation_summary_missing_on_success",
                    "suggestion": "Check the ledger for generated code"
                }
            )
        # Save to ledger
        try:
            saved_path = save_bot_state_to_ledger(state, state.current_user)
        except Exception as e:
            logger.error(f"Failed to save to ledger: {e}")
            saved_path = "Error saving to ledger"
        session_manager_redis.update_session(session_id, {
            "status": "completed",
            "current_stage": "completed"
        })
        # Compile all results safely
        results = {}
        if hasattr(state, 'delta_analyzer_result') and state.delta_analyzer_result:
            results["delta_analyzer"] = {
                "success": getattr(state, 'delta_analyzer_success', False),
                "modifications_count": len(state.delta_analyzer_result.modifications) if hasattr(state.delta_analyzer_result, 'modifications') else 0,
                "new_dependencies": getattr(state.delta_analyzer_result, 'new_dependencies', []),
                "testing_suggestions": getattr(state.delta_analyzer_result, 'testing_suggestions', [])
            }
        if hasattr(state, 'code_generator_result') and state.code_generator_result:
            results["code_generator"] = {
                "success": getattr(state, 'code_generator_success', False),
                "modified_files_count": len(state.code_generator_result.modified_files) if hasattr(state.code_generator_result, 'modified_files') else 0,
                "modified_files": [file_obj.dict() for file_obj in state.code_generator_result.modified_files] if hasattr(state.code_generator_result, "modified_files") else 0,
                "total_modifications": getattr(state.code_generator_result, 'total_modifications', 0)
            }
        if hasattr(state, 'code_validator_result') and state.code_validator_result:
            results["code_validator"] = {
                "success": getattr(state, 'code_validator_success', False),
                "overall_status": getattr(state.code_validator_result, 'overall_status', 'unknown'),
                "quality_score": getattr(state.code_validator_result.validation_summary, 'overall_quality_score', 0),
                "files_validated": getattr(state.code_validator_result.validation_summary, 'total_files', 0)
            }
        return APIResponse(
            session_id=session_id,
            status="completed",
            message="Workflow completed successfully! Code validation passed.",
            workflow_stage="completed",
            data={
                "ledger_path": saved_path,
                "completion_time": datetime.now().isoformat()
            },
            results=results
        )
    # Default processing response
    current_stage = "processing"

    if hasattr(state, 'delta_analyzer_success') and state.delta_analyzer_success:
        current_stage = "code_generation"
    elif hasattr(state, 'master_planner_success') and state.master_planner_success:
        current_stage = "delta_analysis"
    elif hasattr(state, 'core_intent'):
        current_stage = "master_planning"

    return APIResponse(
        session_id=session_id,
        status="processing",
        message="Workflow is processing your request...",
        workflow_stage=current_stage,
        data={
            "current_step": current_stage,
            "progress": "in_progress"
        }
    )

def get_waiting_for_input_response(waiting_resp_dict, waiting_message, double_line):
    waiting_for_input_str = waiting_message + double_line
    waiting_for_input_str += "# Files to modify " + double_line

    for file in waiting_resp_dict["files_to_modify"]:
        waiting_for_input_str += "File path : " + file["file_path"] + double_line
        waiting_for_input_str += "Reason : " + file["analysis"]["reason"] + double_line
    waiting_for_input_str += "Do u want to proceed with the suggestions of the master planner ? if yes , just say yes. if no, provide addditional information on what is new requirement like \"no. requirement\" as format "
    return waiting_for_input_str

def clean_completed_output(code_gen_response_dict, double_line):
    code_gen_response = ""

    if code_gen_response_dict["total_modification"] == 0:

        config_info_dict = ast.literal_eval(code_gen_response_dict["modified_files"][0]["modified_content"])[0]

        code_gen_response += "File name : " + config_info_dict["file_path"] + double_line 
        code_gen_response += "File content :" + "\n```json\n"
        updated_config_str = json.dumps({"sequence": config_info_dict['updated_config']},indent=4)
        updated_config_str = updated_config_str.replace("\\n", '\n')
        updated_config_str = updated_config_str.replace("\\", "")
        code_gen_response += updated_config_str + double_line + "\n```" + double_line
    else:
        for file_detail in code_gen_response_dict["modified_files"]:
            code_gen_response += "File name : " + file_detail["file_path"] + double_line
            code_gen_response += "File content : " + "\n```diff\n"
            code_gen_response += file_detail["modified_content"] + double_line
            code_gen_response += "\n```" + double_line
    return code_gen_response

def get_specific_response_for_display(response):
    try:
        double_line = "\n\n"
        if response.status == "waiting_for_input":
            response_str = get_waiting_for_input_response(response.data, response.message, double_line)
        elif response.status == "completed":
            response_str = clean_completed_output(response_str.results["code_generator"], double_line)
        elif response.status == "query_refinement_needed":
            response_str = response.message + double_line
            response_str += ", ".join(response.data["suggestions"]) + double_line
            response_str += "Please provide a more specific query based on the suggestions provided above"
        elif response.status == "processing" :
            response_str = "Please restart the session again ask the same query with little more information and clearity"
        else:
            response_str = "Please restart the session again ask the same query with little more information and clearity"
        return response_str
    except Exception as e:
        return e

def convert_datetime_to_str(data: dict) -> dict:
    result = {}
    for k,v in data.items():
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        else:
            result[k] = v 
    return result

def get_streaming_display(node_name, node_state):
    display_resp = ""
    if node_name == "communication_node":
        display_resp = "Communication agent is thinking. \n\n"
        if "core_intent" in node_state:
                display_resp += f"- Core intent : {node_state["core_intent"]}\n\n"
        if "context_notes" in node_state:
                display_resp += f"- Context notes : {node_state["context_notes"]}\n\n"
        if "communication_success" in node_state:
                display_resp += f"- Communication success : {node_state["communication_success"]}\n\n"
        return display_resp
    elif node_name == "query_enhancement_node":
        display_resp  += "Query Enhancement agent is thinking. \n\n"
        if "developer_task" in node_state:
            display_resp += f"- Developer task : {node_state["developer_task"]}\n\n"
        if "enhancement_success" in node_state:
                display_resp += f"- Enhancement success : {node_state["enhancement_success"]}\n\n"
        return display_resp
    elif node_name == "master_planner_node":
        display_resp  += "Master Planner Agent is thinking. \n\n"
        if "change_type" in node_state:
            display_resp += f"- Change type : {node_state["change_type"]}\n\n"
        if "master_planner_message" in node_state:
                display_resp += f"- Master planner message : {node_state["master_planner_message"]}\n\n"
        if "master_planner_success" in node_state:
                display_resp += f"- Master planner success : {node_state["master_planner_success"]}\n\n"        
        return display_resp
    elif node_name == "delta_analyzer_node":
        display_resp  += "Delta Analyzer Agent is thinking. \n\n"
        if "delta_analyzer_success" in node_state:
                display_resp += f"- Delta analyzer success : {node_state["delta_analyzer_success"]}\n\n"  
        display_resp += "- The graph has procceeded to code generator node for code generation"              
        return display_resp
    elif node_name == "code_generator_node":
        display_resp  += "Code generator agent is thinking. \n\n"
        if "code_generator_success" in node_state:
                display_resp += f"- Code generator success : {node_state["code_generator_success"]}\n\n"  
        display_resp += "- The graph has procceeded to code validator node for code validation"              
        return display_resp
    elif node_name == "code_validator_node":
        display_resp  += "Code validator agent is thinking. \n\n"
        if "code_validator_success" in node_state:
                display_resp += f"- Code validator success : {node_state["code_validator_success"]}\n\n"               
        return display_resp
    return node_state

@app.get("/")
async def root():
    return {"status" : "OK"}

@app.get("/liveness")
async def liveness():
    return {"status" : "alive"}

@app.get("/readiness")
async def readiness():
    return {"status" : "ready"}
            
@app.post("/chat")
async def chat_endpoint(data: dict):
    try:
        # Handle session creation or retrieval
        message = data["messages"][-1]["content"]
        session_id = data["config"]["chat_Id"]

        user_content_list = [ele for ele in data["messages"]][0::2]
        user_queries = [ele["content"].strip() for ele in user_content_list]

        session = session_manager_redis.get_session(session_id)

        if session is None:
            # Create new session with optional user_id and history loading
            query_type = "initial"
            session_manager_redis.update_history(session_id, user_queries[:-1])
            session = session_manager_redis.create_session(
                session_id=session_id
            )
            is_new_session = True
            logger.info(f"Created new session: {session_id}")
        else:
            query_type = "intermediate"
            is_new_session = False
            
    
        current_state = BotStateSchema(**session["state"])

        # Handle different query types
        if query_type == "initial" or is_new_session:
            # Start new conversation
            session_manager_redis.add_to_session_history(current_state.current_user, message)
            current_state.user_history.append(message)
            current_state.latest_query = message
            logger.info(f"Initial query processed: {query.message[:50]}...")
        elif query_type == "intermediate":
                # Handle query refinement - RESTART WORKFLOW WITH NEW QUERY
            if session["current_stage"] == "query_refinement_needed":
                    # Save refined query to history
                session_manager_redis.add_to_session_history(current_state.current_user, message)
                current_state.user_history.append(message)
                # Reset state for new workflow run
                current_state = BotStateSchema(
                    latest_query=message,
                    user_history=current_state.user_history,
                    current_user=current_state.current_user
                )
                session_manager_redis.update_session(session_id, {
                    "current_stage": "restarting_with_refined_query",
                    "waiting_for_input": False
                })

                session = session_manager_redis.get_session(session_id)
                # Handle master planner approval - SAFE VERSION
            elif session["current_stage"] == "completed":
                session_manager_redis.add_to_session_history(current_state.current_user, message)
                current_state.user_history.append(message)

                original_queries = ", ".join(current_state.user_history[:-1])
                enhanced_task = f"{original_queries}\n\n --- the above are the previously used user_queries. ---"
                enhanced_task+= f"\n Additional requirements: {message}"
                enhanced_task+= "\n\n Please create a new approach considering this feedback."

                preserved_history = current_state.user_history.copy()
                preserved_user = current_state.current_user

                fresh_state_dict = {
                                    "latest_query": enhanced_task,
                                    "developer_task": enhanced_task,
                                    "user_history": preserved_history,
                                    "current_user": preserved_user
                                }
                
                fresh_state = BotStateSchema(**fresh_state_dict)

                session_manager_redis.update_session_state(session_id,fresh_state.dict())

                session_manager_redis.update_session(session_id, {
                                    "waiting_for_input": False,
                                    "current_stage": "follow_up_query",
                                    "status" : "active"
                                })
                
                current_state = fresh_state
            else:
                if "yes" in message.lower():
                    current_state.master_planner_approved = True
                    session_manager_redis.update_session(
                        session_id,
                        {
                            "waiting_for_input" : False,
                            "current_stage" : "continuing_after_approval"
                        }
                    )
                else:
                    current_state.master_planner_approved = False

                    rejection_reason = "i want one more task"
                    additional_requirements = get_additional_requirements(message)

                    rejection_message = "Rejection the master planner proposal."

                    if rejection_reason:
                            rejection_message += f" Reason : {rejection_reason}"
                    if additional_requirements:
                            rejection_message += f" Additional requirements : {additional_requirements}"

                    session_manager_redis.add_to_session_history(current_state.current_user, rejection_message)
                    current_state.user_history.append(rejection_message)

                    original_query = current_state.latest_query
                    enhanced_task += f"{original_query} \n\n--- USER FEEDBACK ---"
                    if rejection_reason:
                        enhanced_task += f"\nRejection reason : {rejection_reason}"
                    if additional_requirements:
                        enhanced_task += f"\n Additional requirements : {additional_requirements}"
                    enhanced_task += "\n\n Please create a new approach considering this feedback."

                    try:
                        preserved_history = current_state.user_history.copy()
                        preserved_user = current_state.current_user

                        fresh_state_dict = {
                                    "latest_query": enhanced_task,
                                    "developer_task": enhanced_task,
                                    "user_history": preserved_history,
                                    "current_user": preserved_user
                                }
                        fresh_state = BotStateSchema(**fresh_state_dict)

                        session_manager_redis.update_session_state(session_id, fresh_state.dict())
                        session_manager_redis.update_session(session_id, {
                            "waiting_for_input" : False,
                            "current_stage" : "restarting_after_rejection"
                        })

                        current_state = fresh_state


                                         # Create a completely fresh state to avoid validation err
                    except Exception as e:
                        logger.error(f"Error creating fresh state: {e}")
                        # Fallback: try selective reset approach
                        logger.info("Falling back to selective reset approach")
                        # Update the tasks first
                        current_state.developer_task = enhanced_task
                        current_state.latest_query = enhanced_task
                        # Safe reset approach - only reset what we can safely reset
                        safe_reset_fields = [
                            'master_planner_success',
                            'master_planner_approved',
                            'delta_analyzer_success',
                            'code_generator_success',
                            'code_validator_success',
                            'communication_success',
                            'enhancement_success',
                            'is_satisfied'
                        ]
                        for field in safe_reset_fields:
                            if hasattr(current_state, field):
                                if 'success' in field or field == 'is_satisfied':
                                    setattr(current_state, field, False)
                                elif 'approved' in field:
                                    setattr(current_state, field, None)
                        # Handle result fields that might cause validation errors
                        result_fields = [
                            'master_planner_result',
                            'delta_analyzer_result',
                            'code_generator_result',
                            'code_validator_result'
                        ]
                        for field in result_fields:
                            if hasattr(current_state, field):
                                try:
                                    # Try setting to empty list first
                                    setattr(current_state, field, [])
                                except:
                                    try:
                                        # If that fails, try None
                                        setattr(current_state, field, None)
                                    except:
                                        # If both fail, leave it as is
                                        logger.warning(f"Could not reset {field}, leaving as is")
                        # Reset other state fields safely
                        optional_reset_fields = ['core_intent', 'context_notes', 'suggestions']
                        for field in optional_reset_fields:
                            if hasattr(current_state, field):
                                try:
                                    setattr(current_state, field, None)
                                except:
                                    logger.warning(f"Could not reset {field}")
                        session_manager_redis.update_session(session_id, {
                            "waiting_for_input": False,
                            "current_stage": "restarting_after_rejection"
                        })
        session = session_manager_redis.get_session(session_id)
        should_execute_workflow = not session.get("waiting_for_input", False)

    
        if should_execute_workflow:
            try:
                logger.info("Starting/Continuing workflow execution...")
                logger.info(f"Current state before workflow:")
                logger.info(f"  - master_planner_approved: {getattr(current_state, 'master_planner_approved', None)}")
                logger.info(f"  - master_planner_success: {getattr(current_state, 'master_planner_success', False)}")

                async def event_stream():
                    try:
                        value = {'index': 0 ,'finish_reason' : None , 'delta' : {'role': 'assistant'}, 'usage': None}
                        payload = {'choices' : [value], "usage": value.get("usage")}
                        yield f"data : {json.dumps(payload)}\n\n"

                        async for step_state in workflow_handler.graph.astream(current_state.dict()):
                            key_name = list(step_state.keys())[0]
                            state = list(step_state.values())[0]

                            display_resp = get_streaming_display(key_name, state)

                            try:
                                value = {'index': 0 ,'finish_reason' : None , 'delta' : {'content': display_resp}, 'usage': None}
                            except:
                                value = {'index': 0 ,'finish_reason' : None , 'delta' : {'content': json.dumps(display_resp)}, 'usage': None}
                            
                            payload = {'choices' : [value], "usage": value.get("usage")}
                            yield f"data: {json.dumps(payload)}\n\n"
                        
                        final_state = BotStateSchema(**state)
                        # Update session with new state
                        session_manager_redis.update_session_state(session_id, final_state.dict())
                        session = session_manager_redis.get_session(session_id)
                        print(session)
                        
                        # Generate response based on final state
                        response = await _generate_response(session_id, final_state, session)
                        response_str = get_specific_response_for_display(response)

                        chunk_size = 100
                        for i in range(0, len(response_str), chunk_size):
                            chunk = response_str[i:i+chunk_size]
                            value = {'index': 0 ,'finish_reason' : None , 'delta' : {'content': chunk}, 'usage': None}
                            payload = {'choices' : [value], "usage": value.get("usage")}
                            yield f"data : {json.dumps(payload)}\n\n"
                            await asyncio.sleep(0.1)
                        
                        value = {'index': 0 ,'finish_reason' : 'stop' , 'delta' : {}, 'usage': None}
                        payload = {'choices' : [value], "usage": value.get("usage")}
                        yield f"data : {json.dumps(payload)}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error' : str(e)})}\n\n"
                return StreamingResponse(event_stream(), media_type='text/event-stream')

            except Exception as e:
                logger.error(f"Workflow execution failed: {e}", exc_info=True)
                async def error_stream():
                    value = {'index': 0 ,'finish_reason' : None , 'delta' : {'role': "assistant"}}
                    payload = {'choices' : [value], "usage": value.get("usage")}
                    yield f"data : {json.dumps(payload)}\n\n"

                    response_str =  json.dumps(APIResponse(
                        session_id=session_id,
                        status="error",
                        message=f"Workflow execution failed: {str(e)}",
                        workflow_stage="error",
                        data={
                            "error_type": type(e).__name__,
                            "suggestion": "Please try again or contact support"
                        }
                    ).dict())
                    value = {'index': 0 ,'finish_reason' : f"\n\nError : {response_str}!" , 'delta' : {}, 'usage' : None}
                    payload = {'choices' : [value], "usage": value.get("usage")}
                    yield f"data : {json.dumps(payload)}\n\n"
                return StreamingResponse(error_stream(), media_type="text/event-stream")    
        else:
            # We're waiting for input, return current state
            async def processing_stream():
                value = {'index': 0 ,'finish_reason' : None , 'delta' : {'role': "assistant"}}
                payload = {'choices' : [value], "usage": value.get("usage")}
                yield f"data : {json.dumps(payload)}\n\n"

                response_str = "Wait for the request to complete. currently not expecting input"
                value = {'index': 0 ,'finish_reason' : f"\n\nError : {response_str}!" , 'delta' : {}, 'usage' : None}
                payload = {'choices' : [value], "usage": value.get("usage")}
                yield f"data : {json.dumps(payload)}\n\n"
            return StreamingResponse(processing_stream(), media_type="text/event-stream")      

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        async def error_stream():
            value = {'index': 0 ,'finish_reason' : None , 'delta' : {'role': "assistent"}}
            payload = {'choices' : [value], "usage": value.get("usage")}
            yield f"data : {json.dumps(payload)}\n\n"

            response_str =  json.dumps(APIResponse(
            session_id=session_id if session_id else "unknown",
            status="error",
            message=f"An error occurred: {str(e)}",
            data={
                "error_type": type(e).__name__
            }
        ).dict())
            value = {'index': 0 ,'finish_reason' : f"\n\nError : {response_str}!" , 'delta' : {}, 'usage' : None}
            payload = {'choices' : [value], "usage": value.get("usage")}
            yield f"data : {json.dumps(payload)}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")    
    

# Additional endpoints for session management
@app.get("/session/{session_id}/status", response_model=SessionInfo)
async def get_session_status(session_id: str):
    session = session_manager_redis.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(
        session_id=session_id,
        status=session["status"],
        last_activity=session["last_activity"],
        current_stage=session["current_stage"],
        query_count=len(session["state"].get("user_history", []))
    )
@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return session_manager_redis.get_all_sessions()

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    return session_manager_redis.delete_session(session_id)

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    session = session_manager_redis.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    state = session["state"]
    return {
        "session_id": session_id,
        "history": state.get("user_history", []),
        "current_task": state.get("developer_task", ""),
        "workflow_stage": session["current_stage"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_manager_redis.get_all_sessions()['sessions']),
        "redis_available": redis_client is not None
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting LangGraph Interactive Code Assistant API")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down LangGraph Interactive Code Assistant API")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  
 