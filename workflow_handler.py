# workflow_handler.py
from threading import Lock
from langgraph.graph import StateGraph
import logging

logger = logging.getLogger(__name__)
graph_lock = Lock()

# Import all your existing functions (keeping them exactly as they are)
from final_flow import (
    communication_node, query_enhancement_node, master_planner_node,delta_analyzer_node, code_generator_node,
    code_validator_node, should_proceed_to_master_planner,
    should_proceed_to_delta_analyzer, should_proceed_to_code_generator,
    should_proceed_to_code_validator, should_end_after_validation
)

class WorkflowHandler:
    def __init__(self):
        self.graph = self._build_graph()
        logger.info("Workflow Handler initialized with LangGraph")
    
    def _build_graph(self):
        """Build the LangGraph workflow (exactly as your original)"""
        builder = StateGraph(dict)
        
        # Add all nodes (same as your original)
        builder.add_node("communication_node", communication_node)
        builder.add_node("query_enhancement_node", query_enhancement_node)
        builder.add_node("master_planner_node", master_planner_node)
        builder.add_node("delta_analyzer_node", delta_analyzer_node)
        builder.add_node("code_generator_node", code_generator_node)
        builder.add_node("code_validator_node", code_validator_node)
        
        # Set entry point
        builder.set_entry_point("communication_node")
        
        # Add edges (exactly as your original)
        builder.add_edge("communication_node", "query_enhancement_node")
        
        builder.add_conditional_edges(
            "query_enhancement_node",
            should_proceed_to_master_planner,  # This is your existing function
            {
                "master_planner": "master_planner_node",
                "__end__": "__end__"  # This is where query refinement failures go
            }
        )
        
        builder.add_conditional_edges(
            "master_planner_node",
            should_proceed_to_delta_analyzer,
            {
                "delta_analyzer": "delta_analyzer_node",
                "restart_from_communication": "communication_node",
                "__end__": "__end__"
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
        
        builder.add_conditional_edges(
            "code_validator_node",
            should_end_after_validation,
            {
                "__end__": "__end__",
                "master_planner": "master_planner_node"
            }
        )
        
        compiled_graph = builder.compile()
        logger.info("LangGraph workflow compiled successfully")
        return compiled_graph
    
    # async def execute_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
    #     """Execute the workflow with thread safety"""
    #     try:
    #         logger.info(f"Executing workflow for query: {state.get('latest_query', 'Unknown')[:50]}...")
            
    #         # Ensure state is properly formatted
    #         state_obj = ensure_state_schema(state)
            
    #         with graph_lock:
    #             result = await self.graph.ainvoke(state_obj.dict())
            
    #         logger.info("Workflow execution completed successfully")
    #         return result
            
        # except Exception as e:
        #     logger.error(f"Workflow execution error: {e}")
        #     raise

# Global workflow handler instance
workflow_handler = WorkflowHandler()