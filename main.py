from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
import uvicorn 
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
import asyncio
from enum import Enum
import json
import logging
from datetime import datetime
from pathlib import Path
# Import your existing modules
from agents import (CommunicationAgent, QueryRephraserAgent, MasterPlannerAgent,
                   DeltaAnalyzerAgent, CodeGeneratorAgent, CodeValidatorAgent)
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
# Data Models for API
class QueryType(str, Enum):
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    FEEDBACK = "feedback"

class UserQuery(BaseModel):
    session_id: Optional[str] = None
    query_type: QueryType
    message: str
    user_input: Optional[Dict[str, Any]] = None
    feedback_type: Optional[str] = None
    # New fields for user management
    user_id: Optional[str] = None  # Specific user ID to use
    load_history_from: Optional[str] = None  # User ID to load history from

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
    user_id: str
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
from session_manager_redis import RedisSessionManager
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
        RedisSessionManager.update_session(session_id, {
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
        RedisSessionManager.update_session(session_id, {
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
        RedisSessionManager.update_session(session_id, {
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
        RedisSessionManager.update_session(session_id, {
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
@app.post("/chat", response_model=APIResponse)
async def chat_endpoint(query: UserQuery):
    session_id = None
    try:
        # Handle session creation or retrieval
        if query.session_id is None:
            # Create new session with optional user_id and history loading
            session_id = RedisSessionManager.create_session(
                user_id=query.user_id,
                load_history_from=query.load_history_from
            )
            is_new_session = True
            logger.info(f"Created new session: {session_id}")
        else:
            session_id = query.session_id
            is_new_session = False
            session = RedisSessionManager.get_session(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found or expired")
        session = RedisSessionManager.get_session(session_id)
        current_state = BotStateSchema(**session["state"])
        logger.info(f"Processing query type: {query.query_type} for session: {session_id}")
        # Handle different query types
        if query.query_type == QueryType.INITIAL or is_new_session:
            # Start new conversation
            RedisSessionManager.add_to_user_history(current_state.current_user, query.message)
            current_state.user_history.append(query.message)
            current_state.latest_query = query.message
            logger.info(f"Initial query processed: {query.message[:50]}...")
        elif query.query_type == QueryType.INTERMEDIATE:
            # Handle user input for conditional nodes
            if query.user_input:
                logger.info(f"Processing intermediate input: {list(query.user_input.keys())}")
                # Handle query refinement - RESTART WORKFLOW WITH NEW QUERY
                if query.user_input.get("refined_query"):
                    refined_query = query.user_input["refined_query"]
                    logger.info(f"Query refinement received: {refined_query[:50]}...")
                    # Save refined query to history
                    RedisSessionManager.add_to_user_history(current_state.current_user, refined_query)
                    current_state.user_history.append(refined_query)
                    # Reset state for new workflow run
                    current_state = BotStateSchema(
                        latest_query=refined_query,
                        user_history=current_state.user_history,
                        current_user=current_state.current_user
                    )
                    RedisSessionManager.update_session(session_id, {
                        "current_stage": "restarting_with_refined_query",
                        "waiting_for_input": False
                    })
                # Handle master planner approval - SAFE VERSION
                elif "master_planner_approved" in query.user_input:
                    approval = query.user_input["master_planner_approved"]
                    logger.info(f"Master planner approval received: {approval}")
                    if isinstance(approval, str):
                        if approval.lower() in ["yes", "true", "approve", "y", "1"]:
                            current_state.master_planner_approved = True
                            logger.info("Master planner approved by user - continuing workflow")
                            RedisSessionManager.update_session(session_id, {
                                "waiting_for_input": False,
                                "current_stage": "continuing_after_approval"
                            })
                        elif approval.lower() in ["no", "false", "reject", "n", "0"]:
                            current_state.master_planner_approved = False
                            logger.info("Master planner rejected by user - restarting process")
                            # Get rejection reason and additional requirements
                            rejection_reason = query.user_input.get("rejection_reason", "")
                            additional_requirements = query.user_input.get("additional_requirements", "")
                            # Create rejection message for history
                            rejection_message = "Rejected the master planner proposal."
                            if rejection_reason:
                                rejection_message += f" Reason: {rejection_reason}"
                            if additional_requirements:
                                rejection_message += f" Additional requirements: {additional_requirements}"
                            # Add rejection to user history
                            RedisSessionManager.add_to_user_history(current_state.current_user, rejection_message)
                            current_state.user_history.append(rejection_message)
                            # Recreate developer task with feedback
                            original_query = current_state.latest_query
                            enhanced_task = f"{original_query}\n\n--- USER FEEDBACK ---"
                            if rejection_reason:
                                enhanced_task += f"\nRejection reason: {rejection_reason}"
                            if additional_requirements:
                                enhanced_task += f"\nAdditional requirements: {additional_requirements}"
                            enhanced_task += "\n\nPlease create a new approach considering this feedback."
                            # Create a completely fresh state to avoid validation errors
                            try:
                                # Get the current user history and user info
                                preserved_history = current_state.user_history.copy()
                                preserved_user = current_state.current_user
                                # Create minimal fresh state with only required fields
                                fresh_state_dict = {
                                    "latest_query": enhanced_task,
                                    "developer_task": enhanced_task,
                                    "user_history": preserved_history,
                                    "current_user": preserved_user
                                }
                                # Create new BotStateSchema instance
                                fresh_state = BotStateSchema(**fresh_state_dict)
                                # Update session with completely fresh state
                                RedisSessionManager.update_session_state(session_id, fresh_state.dict())
                                RedisSessionManager.update_session(session_id, {
                                    "waiting_for_input": False,
                                    "current_stage": "restarting_after_rejection"
                                })
                                logger.info("Created fresh state for restart - all validation errors avoided")
                                # Update current_state reference for immediate use
                                current_state = fresh_state
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
                                RedisSessionManager.update_session(session_id, {
                                    "waiting_for_input": False,
                                    "current_stage": "restarting_after_rejection"
                                })
                    else:
                        # Handle direct boolean values
                        current_state.master_planner_approved = bool(approval)
                        logger.info(f"Master planner approval set to: {current_state.master_planner_approved}")
                        if current_state.master_planner_approved:
                            RedisSessionManager.update_session(session_id, {
                                "waiting_for_input": False,
                                "current_stage": "continuing_after_approval"
                            })
                        else:
                            # Simple rejection without additional feedback
                            rejection_message = "Rejected the master planner proposal without additional feedback."
                            RedisSessionManager.add_to_user_history(current_state.current_user, rejection_message)
                            current_state.user_history.append(rejection_message)
                            # Use the same fresh state approach for simple rejection
                            try:
                                preserved_history = current_state.user_history.copy()
                                preserved_user = current_state.current_user
                                current_task = getattr(current_state, 'developer_task', current_state.latest_query)
                                fresh_state_dict = {
                                    "latest_query": current_state.latest_query,
                                    "developer_task": current_task,
                                    "user_history": preserved_history,
                                    "current_user": preserved_user
                                }
                                fresh_state = BotStateSchema(**fresh_state_dict)
                                RedisSessionManager.update_session_state(session_id, fresh_state.dict())
                                RedisSessionManager.update_session(session_id, {
                                    "waiting_for_input": False,
                                    "current_stage": "restarting_after_rejection"
                                })
                                current_state = fresh_state
                            except Exception as e:
                                logger.error(f"Error in simple rejection fresh state creation: {e}")
                                # Keep current state as is and just update session
                                RedisSessionManager.update_session(session_id, {
                                    "waiting_for_input": False,
                                    "current_stage": "restarting_after_rejection"
                                })
                # Handle validation feedback
                elif query.user_input.get("validation_feedback"):
                    feedback = query.user_input["validation_feedback"]
                    logger.info(f"Validation feedback received: {feedback[:50]}...")
                    current_state.developer_task = f"{getattr(current_state, 'developer_task', '')}\n\nValidation feedback:\n{feedback}"
                    RedisSessionManager.add_to_user_history(current_state.current_user, feedback)
        elif query.query_type == QueryType.FEEDBACK:
            # Handle specific feedback (new query, clarifications, etc.)
            logger.info(f"Processing feedback type: {query.feedback_type}")
            if query.feedback_type == "new_requirements":
                original_task = getattr(current_state, 'developer_task', '')
                updated_task = f"{original_task}\n\n--- ADDITIONAL REQUIREMENTS ---\n{query.message}"
                current_state.developer_task = updated_task
                current_state.latest_query = query.message
                RedisSessionManager.add_to_user_history(current_state.current_user, query.message)
        # Execute workflow only if we're not waiting for input
        session = RedisSessionManager.get_session(session_id)  # Get updated session
        should_execute_workflow = not session.get("waiting_for_input", False)
        if should_execute_workflow:
            try:
                logger.info("Starting/Continuing workflow execution...")
                logger.info(f"Current state before workflow:")
                logger.info(f"  - master_planner_approved: {getattr(current_state, 'master_planner_approved', None)}")
                logger.info(f"  - master_planner_success: {getattr(current_state, 'master_planner_success', False)}")
                result_state = await workflow_handler.execute_workflow(current_state.dict())
                final_state = BotStateSchema(**result_state)
                # Log workflow progression
                logger.info(f"Workflow completed. Final state stages:")
                logger.info(f"  - Master planner: {getattr(final_state, 'master_planner_success', False)}")
                logger.info(f"  - Delta analyzer: {getattr(final_state, 'delta_analyzer_success', False)}")
                logger.info(f"  - Code generator: {getattr(final_state, 'code_generator_success', False)}")
                logger.info(f"  - Code validator: {getattr(final_state, 'code_validator_success', False)}")
                # Update session with new state
                RedisSessionManager.update_session_state(session_id, final_state.dict())
                # Generate response based on final state
                response = await _generate_response(session_id, final_state, RedisSessionManager.get_session(session_id))
                return response
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}", exc_info=True)
                return APIResponse(
                    session_id=session_id,
                    status="error",
                    message=f"Workflow execution failed: {str(e)}",
                    workflow_stage="error",
                    data={
                        "error_type": type(e).__name__,
                        "suggestion": "Please try again or contact support"
                    }
                )
        else:
            # We're waiting for input, return current state
            logger.info("Waiting for user input, not executing workflow")
            response = await _generate_response(session_id, current_state, session)
            return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return APIResponse(
            session_id=session_id if session_id else "unknown",
            status="error",
            message=f"An error occurred: {str(e)}",
            data={
                "error_type": type(e).__name__
            }
        )
# Additional endpoints for session management
@app.get("/session/{session_id}/status", response_model=SessionInfo)
async def get_session_status(session_id: str):
    session = RedisSessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(
        session_id=session_id,
        status=session["status"],
        last_activity=session["last_activity"],
        current_stage=session["current_stage"],
        user_id=session["user_id"],
        query_count=len(session["state"].get("user_history", []))
    )
@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return RedisSessionManager.get_all_sessions()
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in RedisSessionManager.sessions:
        del RedisSessionManager.sessions[session_id]
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    session = RedisSessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    state = session["state"]
    return {
        "session_id": session_id,
        "user_id": session["user_id"],
        "history": state.get("user_history", []),
        "current_task": state.get("developer_task", ""),
        "workflow_stage": session["current_stage"]
    }
@app.post("/session/{session_id}/clear-history")
async def clear_session_history(session_id: str):
    """Clear conversation history for a session"""
    session = RedisSessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # Clear Redis history
    try:
        from final_flow import clear_user_history
        user_id = session["user_id"]
        clear_user_history(user_id)
    except Exception as e:
        logger.warning(f"Could not clear Redis history: {e}")
    # Reset session state
    session["state"]["user_history"] = []
    RedisSessionManager.update_session(session_id, {"current_stage": "initial"})
    return {"message": "History cleared successfully"}
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(RedisSessionManager.sessions),
        "redis_available": redis_client is not None
    }
@app.get("/users")
async def list_users():
    """List all users with their history information"""
    return RedisSessionManager.get_all_users()
@app.get("/user/{user_id}/history")
async def get_user_history_endpoint(user_id: str):
    """Get history for a specific user"""
    history = RedisSessionManager.get_user_history(user_id)
    return {
        "user_id": user_id,
        "history": history,
        "history_count": len(history)
    }
@app.delete("/user/{user_id}/history")
async def clear_user_history_endpoint(user_id: str):
    """Clear history for a specific user"""
    if user_id in RedisSessionManager.user_histories:
        del RedisSessionManager.user_histories[user_id]
        return {"message": f"History cleared for user {user_id}"}
    else:
        raise HTTPException(status_code=404, detail="User not found")
@app.post("/session/create")
async def create_session_with_options(
    user_id: Optional[str] = None,
    load_history_from: Optional[str] = None
):
    """Create a new session with specific user ID and/or load history from another user"""
    session_id = RedisSessionManager.create_session(
        user_id=user_id,
        load_history_from=load_history_from
    )
    session = RedisSessionManager.get_session(session_id)
    return {
        "session_id": session_id,
        "user_id": session["user_id"],
        "loaded_history_from": session.get("loaded_history_from"),
        "history_count": len(session["state"]["user_history"])
    }
# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting LangGraph Interactive Code Assistant API")
    RedisSessionManager.cleanup_expired_sessions()
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down LangGraph Interactive Code Assistant API")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  
 