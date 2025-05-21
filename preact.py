import dotenv
dotenv.load_dotenv()
import json
import re
import uuid
from typing import Annotated, Dict, List, Sequence, TypedDict, Union, Any, Optional, Callable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import from the new modules
from tools import ToolRegistry, default_registry
from prompts import get_prompt

# Define the Pre-Act agent state
class AgentState(TypedDict):
    """State for the Pre-Act agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Track steps executed for the current user request
    executed_steps: List[Dict]
    # Track if we're in planning mode or execution mode
    current_plan: Optional[List[Dict]]
    # Track if we need to generate final answer
    should_generate_answer: bool


class PreActAgent:
    """
    Pre-Act Agent that implements the multi-step planning approach.
    
    This agent follows the Pre-Act pattern described in the paper:
    "Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents"
    
    The agent generates comprehensive plans with detailed reasoning before 
    taking actions, and refines its plans based on observations from tool executions.
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tool_registry: Optional[ToolRegistry] = None,
        verbose: bool = False
    ):
        """
        Initialize the Pre-Act agent.
        
        Args:
            llm: The language model to use for generating plans and actions
            tool_registry: A tool registry instance with tools the agent can use
            verbose: Whether to print detailed logs during execution
        """
        # Set default LLM if not provided
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o-mini")
        
        # Use default tool registry if none provided
        self.tool_registry = tool_registry or default_registry
        
        # Get tools from the registry
        self.tools = self.tool_registry.get_all_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_names = ", ".join(self.tool_registry.get_tool_names())
        
        # Get the prompt template
        self.prompt_template = get_prompt()
        
        # Set verbosity
        self.verbose = verbose
        
        # Initialize the agent graph
        self.agent = self._build_agent_graph()
    
    def _parse_llm_response(self, response_content: str) -> Dict:
        """
        Parse the LLM's response to extract the action and thought.
        
        Args:
            response_content: The text content of the LLM's response
            
        Returns:
            A dictionary containing the parsed action data and thought
        """
        if self.verbose:
            print(f"Raw LLM response: {response_content}")
            
        result = {
            "action_data": None,
            "thought": "",
            "plan_steps": []
        }
        
        try:
            # Extract the JSON blob using regex
            json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
            json_match = re.search(json_pattern, response_content)
            
            if json_match:
                action_json = json_match.group(1).strip()
                result["action_data"] = json.loads(action_json)
            else:
                raise ValueError("Could not find JSON in response")
                
            # Extract the thought using regex
            thought_pattern = r'^(.*?)(?=Action:)'
            thought_match = re.search(thought_pattern, response_content, re.DOTALL | re.MULTILINE)
            result["thought"] = thought_match.group(1).strip() if thought_match else ""
            
            # Parse plan steps from the thought
            if "Next Steps (Plan):" in result["thought"]:
                plan_section = result["thought"].split("Next Steps (Plan):")[1].strip()
                step_lines = plan_section.split("\n")
                for line in step_lines:
                    if line.strip() and line[0].isdigit() and "." in line:
                        result["plan_steps"].append({"step": line.strip()})
                        
        except Exception as e:
            if self.verbose:
                print(f"Error parsing LLM response: {e}")
                print(f"Response content: {response_content}")
            
            # Default to a final answer if parsing fails
            result["action_data"] = {
                "action": "Final Answer",
                "action_input": "I apologize, but I encountered an error processing your request. Please try again."
            }
            
        return result
    
    def _generate_plan_node(self, state: AgentState, config: RunnableConfig) -> Dict:
        """
        Node for generating a multi-step plan and determining the first action.
        
        Args:
            state: The current agent state
            config: Configuration for the runnable
            
        Returns:
            Updated agent state
        """
        # Create the system message with tools and prompt
        system_message = SystemMessage(
            content=self.prompt_template.format(
                tools="\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                tool_names=self.tool_names,
            )
        )

        # Prepare current messages
        current_messages = [system_message] + state["messages"]
        
        # Invoke the model to get a response
        response = self.llm.invoke(current_messages, config)
        
        # Parse the response
        parsed_response = self._parse_llm_response(response.content)
        action_data = parsed_response["action_data"]
        thought = parsed_response["thought"]
        plan_steps = parsed_response["plan_steps"]
        
        # Determine if we need to generate an answer or execute a tool
        if action_data["action"] == "Final Answer":
            return {
                "messages": [AIMessage(content=action_data["action_input"])],
                "executed_steps": state.get("executed_steps", []),
                "current_plan": plan_steps,
                "should_generate_answer": True
            }
        else:
            # Extract the tool name and input
            tool_name = action_data["action"]
            tool_input = action_data["action_input"]
            
            # Create a unique ID for the tool call
            tool_id = f"call_{uuid.uuid4().hex[:10]}"
            
            # Create a custom AIMessage
            ai_message = AIMessage(content=thought)
            
            # Add tool_calls attribute manually
            ai_message.tool_calls = [{
                "name": tool_name,
                "args": tool_input,
                "id": tool_id
            }]
            
            return {
                "messages": [ai_message],
                "executed_steps": state.get("executed_steps", []),
                "current_plan": plan_steps,
                "should_generate_answer": False
            }
    
    def _execute_tools_node(self, state: AgentState) -> Dict:
        """
        Node for executing tools called by the agent.
        
        Args:
            state: The current agent state
            
        Returns:
            Updated agent state with tool execution results
        """
        outputs = []
        executed_steps = state.get("executed_steps", [])
        new_executed_steps = []
        
        # Get the most recent message with tool calls
        last_message = state["messages"][-1]
        
        # Execute each tool call
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if self.verbose:
                print(f"Executing tool: {tool_name} with args: {tool_args}")
            
            # Execute the tool and get the result
            try:
                # Get the tool from the registry
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Unknown tool: {tool_name}")
                    
                tool_result = tool.invoke(tool_args)
                
                # Track the executed step
                new_executed_steps.append({
                    "tool": tool_name,
                    "input": tool_args,
                    "output": tool_result
                })
                
                # Create a tool message with the result
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result),
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                )
                
                if self.verbose:
                    print(f"Tool result: {tool_result}")
                    
            except Exception as e:
                # Handle errors in tool execution
                error_message = f"Error executing tool {tool_name}: {str(e)}"
                outputs.append(
                    ToolMessage(
                        content=error_message,
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                )
                new_executed_steps.append({
                    "tool": tool_name,
                    "input": tool_args,
                    "output": error_message,
                    "error": True
                })
                
                if self.verbose:
                    print(f"Tool error: {error_message}")
        
        # Update the executed steps
        executed_steps.extend(new_executed_steps)
        
        return {
            "messages": outputs,
            "executed_steps": executed_steps,
            "current_plan": state.get("current_plan", []),
            "should_generate_answer": state.get("should_generate_answer", False)
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if we should continue executing tools or generate a final answer.
        
        Args:
            state: The current agent state
            
        Returns:
            The next edge to follow in the graph
        """
        if state.get("should_generate_answer", False):
            return "end"
        
        # Check if the last message was a tool response
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            # After receiving tool output, go back to planning
            return "continue_planning"
        
        # Check if the last message has tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # If there are tool calls, execute them
            return "execute_tools"
        
        # Default to ending the graph
        return "end"
    
    def _build_agent_graph(self):
        """
        Build and compile the agent graph.
        
        Returns:
            The compiled agent graph
        """
        # Initialize the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self._generate_plan_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        
        # Set entry point
        workflow.set_entry_point("plan")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "plan",
            self._should_continue,
            {
                "execute_tools": "execute_tools",
                "end": END,
            }
        )
        
        workflow.add_conditional_edges(
            "execute_tools",
            self._should_continue,
            {
                "continue_planning": "plan",
                "end": END,
            }
        )
        
        # Compile the graph
        return workflow.compile()
    
    def invoke(self, query: str, **kwargs) -> List[BaseMessage]:
        """
        Run the Pre-Act agent with a query.
        
        Args:
            query: The user's query
            **kwargs: Additional configuration parameters
            
        Returns:
            A list of messages representing the conversation
        """
        # Prepare the initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "executed_steps": [],
            "current_plan": None,
            "should_generate_answer": False
        }
        
        # Run the agent
        result = self.agent.invoke(initial_state, **kwargs)
        
        # Return the agent's messages
        return result["messages"]
    
    def stream(self, query: str, **kwargs):
        """
        Stream the Pre-Act agent's execution.
        
        Args:
            query: The user's query
            **kwargs: Additional configuration parameters
            
        Returns:
            An iterator over the agent's execution steps
        """
        # Prepare the initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "executed_steps": [],
            "current_plan": None,
            "should_generate_answer": False
        }
        
        # Stream the agent execution
        return self.agent.stream(initial_state, **kwargs)


def print_conversation(messages: List[BaseMessage]) -> None:
    """
    Helper function to pretty print a conversation.
    
    Args:
        messages: List of messages to print
    """
    for message in messages:
        if isinstance(message, HumanMessage):
            print(f"\n🧑‍💻 Human: {message.content}")
        elif isinstance(message, AIMessage):
            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"\n🤖 AI (Planning):")
                if message.content:
                    print(f"{message.content}")
                print("\nTool Calls:")
                for tool_call in message.tool_calls:
                    print(f"  - {tool_call['name']}: {tool_call['args']}")
            else:
                print(f"\n🤖 AI (Final Answer): {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"\n🛠️ Tool ({message.name}): {message.content}")


# Example usage
if __name__ == "__main__":
    # Create the Pre-Act agent with the default registry
    agent = PreActAgent(
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        tool_registry=default_registry,
        verbose=True
    )
    
    # Run the agent
    query = "What's the weather in San Francisco and the latest news in the United States?"
    result = agent.invoke(query)
    
    # Print the conversation
    print_conversation(result)
