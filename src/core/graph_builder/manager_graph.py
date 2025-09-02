from typing import Dict, Any
from langgraph.graph import StateGraph, END, START

from src.core.nodes.caption_node import caption_node
from src.core.state import ViReAgentState, ViReManagerState, ManagerOutputState
from src.core.nodes.subgraph_node import tool_node, call_agent_node, final_reasoning_node, should_continue, rationale_node
from src.agents.strategies.manager_agent import ManagerAgent

class ManagerGraphBuilder:
    """Builder for the manager agent workflow with VQA tool, knowledge base tools, and object analysis"""
    
    def __init__(self, tools_registry: Dict[str, Any]):
        self.tools_registry = tools_registry
        
    def create_manager_workflow(self):
        """Create the main manager workflow that starts from caption"""
        main = StateGraph(ViReAgentState)
        
        # Add caption node
        main.add_node("caption", caption_node)
        
        # Add manager analyst subgraph
        main.add_node("manager_analyst", self.create_manager_subgraph())
        
        # Add final answer processing node
        main.add_node("final_processing", self.final_processing_node)
        
        # Add edges
        main.add_edge(START, "caption")
        main.add_edge("caption", "manager_analyst")
        main.add_edge("manager_analyst", "final_processing")
        main.add_edge("final_processing", END)
        
        return main.compile()
    
    def final_processing_node(self, state):
        """Process manager agent results into final answer and explanation"""
        # Get the results from manager analyst
        results = state.get("results", [])
        rationales = state.get("rationales", [])
        
        # Extract manager's answer and rationale
        final_answer = ""
        explanation = ""
        
        if results and isinstance(results, list) and results:
            final_answer = results[0].get("Manager", "")
        
        if rationales and isinstance(rationales, list) and rationales:
            explanation = rationales[0].get("Manager", "")
        
        return {
            "final_answer": final_answer,
            "explanation": explanation
        }
        
    def create_manager_subgraph(self):
        """Create subgraph for Manager Analyst with VQA tool, knowledge base tools, and object analysis"""
        manager_analyst = ManagerAgent()
        workflow = StateGraph(ViReManagerState, output=ManagerOutputState)
        
        def agent_node(state, config):
            state["analyst"] = manager_analyst 
            return call_agent_node(state, config, self.tools_registry)
        
        def tools_node(state):
            return tool_node(state, self.tools_registry)
        
        def final_reasoning_with_analyst(state):
            return final_reasoning_node(state)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tools_node)
        workflow.add_node("rationale", rationale_node)
        workflow.add_node("final_reasoning", final_reasoning_with_analyst)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges("agent", should_continue, {
            "continue": "tools",
            "rationale": "rationale"
        })
        
        # Add edges
        workflow.add_edge("tools", "agent")
        workflow.add_edge("rationale", "final_reasoning")
        workflow.add_edge("final_reasoning", END)
        
        return workflow.compile()