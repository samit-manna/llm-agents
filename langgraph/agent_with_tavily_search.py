"""
LangGraph agent with Calculator and Tavily Search tools
"""
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from operator import add

# Load environment variables
load_dotenv()

# Define calculator tool
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
        result = eval(operation)
        print(f"  ✓ Calculator result: {result}")
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

# Define search tool
search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False
)

# Rename for better display
search_tool.name = "web_search"
search_tool.description = "Search the web for current information, news, facts, or any information you don't know. Input should be a search query."

# State
class State(TypedDict):
    messages: Annotated[list, add]

# Agent node
def agent(state: State):
    """Agent that can use tools to answer questions"""
    print("  🤔 Agent thinking...")
    
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    
    # Bind both tools
    llm_with_tools = llm.bind_tools([calculator, search_tool])
    response = llm_with_tools.invoke(state["messages"])
    
    # Check what the agent decided
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_name = response.tool_calls[0]['name']
        print(f"  🔧 Agent decided to use tool: {tool_name}")
    else:
        print("  💬 Agent answering directly (no tool needed)")
    
    return {"messages": [response]}

# Router function
def should_continue(state: State):
    """Check if the agent wants to use a tool"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# Build graph
def create_graph():
    """Create an agent graph with calculator and search tools"""
    graph = StateGraph(State)
    
    # Create tool node with both tools
    tool_node = ToolNode([calculator, search_tool])
    
    # Add nodes
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    
    # Define flow
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "agent")
    
    return graph.compile()

# Run the agent
if __name__ == "__main__":
    agent = create_graph()
    
    print("LangGraph Agent with Calculator + Web Search")
    print("=" * 60)
    
    # Test 1: Math question (uses calculator)
    print("\n🧑 User: What is 456 * 789?")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is 456 * 789?")]
    })
    print(f"🤖 Agent: {result['messages'][-1].content}")
    
    # Test 2: Current information (uses web search)
    print("\n" + "-" * 60)
    print("\n🧑 User: What are the latest developments in AI agents in 2025?")
    result = agent.invoke({
        "messages": [HumanMessage(content="What are the latest developments in AI agents in 2025?")]
    })
    print(f"🤖 Agent: {result['messages'][-1].content}")
    
    # Test 3: General knowledge (no tool needed)
    print("\n" + "-" * 60)
    print("\n🧑 User: What is the capital of Japan?")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is the capital of Japan?")]
    })
    print(f"🤖 Agent: {result['messages'][-1].content}")
    
    # Test 4: Specific search query
    print("\n" + "-" * 60)
    print("\n🧑 User: Search for information about LangGraph framework")
    result = agent.invoke({
        "messages": [HumanMessage(content="Search for information about LangGraph framework")]
    })
    print(f"🤖 Agent: {result['messages'][-1].content}")
    
    print("\n" + "=" * 60)
    print("✓ Multi-tool agent completed!")
    print("\n📊 Check LangSmith to see:")
    print("   - Which tool was used for each question")
    print("   - Full execution trace with timings")
    print("   - Tool inputs and outputs")
    print("\n🔗 https://smith.langchain.com/")
