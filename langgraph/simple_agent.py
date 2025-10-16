"""
Simple LangGraph agent with a tool (calculator)
"""
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from operator import add

# Load environment variables
load_dotenv()

# Define a simple calculator tool
@tool
def calculator(operation: str) -> str:
    """Performs basic math operations. 
    
    Args:
        operation: A math expression like '2 + 2' or '10 * 5'
    
    Returns:
        The result of the calculation
    """
    print(f"  🧮 Executing calculator with: {operation}")
    try:
        # Using eval for simplicity (in production, use a safer parser)
        result = eval(operation)
        print(f"  ✓ Calculator result: {result}")
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


# Define the state - what data flows through the graph
class State(TypedDict):
    messages: Annotated[list, add]  # Messages accumulate in the list

# Define the agent node - decides whether to use tools or respond
def agent(state: State):
    """Agent that can use tools to answer questions"""
    print("  🤔 Agent thinking...")
    
    # Get the LLM with tool binding
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    
    # Bind the calculator tool to the LLM
    llm_with_tools = llm.bind_tools([calculator])
    
    # Get response from LLM
    response = llm_with_tools.invoke(state["messages"])
    
    # Check if tools are being called
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"  🔧 Agent decided to use tool: {response.tool_calls[0]['name']}")
    else:
        print("  💬 Agent answering directly (no tool needed)")
    
    # Return the response (will include tool calls if needed)
    return {"messages": [response]}


# Define should_continue function - decides if we need to call tools
def should_continue(state: State):
    """Check if the agent wants to use a tool"""
    last_message = state["messages"][-1]
    
    # If there are tool calls, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, we're done
    return END


# Build the graph
def create_graph():
    """Create an agent graph with tool support"""
    # Initialize graph
    graph = StateGraph(State)
    
    # Create tool node (handles tool execution)
    tool_node = ToolNode([calculator])
    
    # Add nodes
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    
    # Define the flow
    graph.add_edge(START, "agent")
    
    # Conditional edge: agent decides if it needs tools
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    
    # After using tools, go back to agent
    graph.add_edge("tools", "agent")
    
    # Compile the graph
    return graph.compile()


# Run the agent
if __name__ == "__main__":
    # Create the agent
    agent = create_graph()
    
    # Test it with questions that need the calculator
    print("LangGraph Agent with Calculator Tool")
    print("=" * 50)
    
    # Test 1: Math question
    print("\n🧑 User: What is 123 * 456?")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is 123 * 456?")]
    })
    print(f"🤖 Agent: {result['messages'][-1].content}")
    
    # Test 2: Regular question (no tool needed)
    print("\n" + "-" * 50)
    print("\n🧑 User: What's the capital of France?")
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the capital of France?")]
    })
    print(f"🤖 Agent: {result['messages'][-1].content}")
    
    # Test 3: Complex math
    print("\n" + "-" * 50)
    print("\n🧑 User: Calculate 636387648 / 36735")
    result = agent.invoke({
        "messages": [HumanMessage(content="Calculate 636387648 / 36735")]
    })
    print(f"🤖 Agent: {result['messages'][-1].content}")
    
    print("\n" + "=" * 50)
    print("✓ Agent with tools completed successfully!")
    print("\nCheck LangSmith for trace: https://smith.langchain.com/")
    print("\nNote: You'll see the agent decide when to use the calculator tool!")
