"""
LangGraph Simple ReAct Agent - Day 1 Morning
Modified to work with Azure OpenAI

Installation:
pip install langgraph langchain-openai langchain-core python-dotenv

Run:
python agent.py
"""

import os
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import operator

# ============================================================================
# 🔑 LOAD ENVIRONMENT VARIABLES
# ============================================================================
load_dotenv()

# Azure OpenAI configuration will be loaded from .env file:
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_VERSION
# - AZURE_OPENAI_DEPLOYMENT_NAME

# LangSmith tracing (already configured in .env)
# - LANGCHAIN_TRACING_V2=true
# - LANGCHAIN_API_KEY
# - LANGCHAIN_PROJECT

# ============================================================================
# 🛠️ DEFINE TOOLS
# ============================================================================

@tool
def search_tool(query: str) -> str:
    """Search for information. Use this when you need current information or facts."""
    mock_results = {
        "weather": "Today's weather: Sunny, 75°F (24°C), light breeze",
        "python": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs using graph-based workflows",
        "ai": "Artificial Intelligence (AI) is the simulation of human intelligence by machines",
        "order": "Order system is operational. Use order_lookup_tool with specific order ID.",
    }
    
    query_lower = query.lower()
    for keyword, result in mock_results.items():
        if keyword in query_lower:
            return f"📰 Search Result: {result}"
    
    return f"📰 Search Result: Information about '{query}' - General knowledge query processed."

@tool
def calculator_tool(expression: str) -> str:
    """Calculate mathematical expressions. Input should be a valid math expression like '2+2' or '15*37'."""
    try:
        # Safe evaluation for basic math
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "❌ Error: Only basic math operators (+, -, *, /, parentheses) allowed"
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"🔢 Calculation Result: {expression} = {result}"
    except Exception as e:
        return f"❌ Calculation Error: {str(e)}"

@tool
def order_lookup_tool(order_id: str) -> str:
    """Look up order status by order ID. Use this for order-related queries."""
    orders_db = {
        "12345": {
            "status": "Shipped",
            "date": "Oct 10, 2025",
            "delivery": "Oct 18, 2025",
            "items": "2 items",
            "tracking": "TRK-8372-XYZ"
        },
        "67890": {
            "status": "Processing",
            "date": "Oct 15, 2025",
            "delivery": "Est. Oct 20, 2025",
            "items": "1 item",
            "tracking": "Pending"
        },
        "11111": {
            "status": "Delivered",
            "date": "Oct 8, 2025",
            "delivery": "Oct 12, 2025",
            "items": "3 items",
            "tracking": "Delivered to door"
        }
    }
    
    if order_id in orders_db:
        order = orders_db[order_id]
        return (f"📦 Order #{order_id}:\n"
                f"  Status: {order['status']}\n"
                f"  Order Date: {order['date']}\n"
                f"  Delivery: {order['delivery']}\n"
                f"  Items: {order['items']}\n"
                f"  Tracking: {order['tracking']}")
    
    return f"❌ Order #{order_id} not found. Please check the order ID."

# ============================================================================
# 📊 DEFINE STATE
# ============================================================================

class AgentState(TypedDict):
    """
    The state of our agent conversation.
    - messages: Full conversation history
    - operator.add: Ensures new messages are appended, not replaced
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ============================================================================
# 🤖 CREATE AGENT
# ============================================================================

# Initialize LLM with tools
tools = [search_tool, calculator_tool, order_lookup_tool]

# Use Azure OpenAI instead of regular OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    # temperature=0,  # Some Azure models don't support temperature parameter
)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """
    The agent's reasoning node.
    Takes conversation history, decides whether to use tools or respond.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """
    Routing logic: Check if agent wants to use tools.
    - If tool_calls exist → route to "tools"
    - Otherwise → route to "end"
    """
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"

# ============================================================================
# 🗺️ BUILD THE GRAPH
# ============================================================================

def create_agent():
    """
    Create the agent graph:
    
    START → agent → should_continue?
                        ↓
                    tools (if needed)
                        ↓
                    agent (process tool results)
                        ↓
                    END
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # Set starting point
    workflow.set_entry_point("agent")
    
    # Add conditional routing from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools execute, go back to agent
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# ============================================================================
# 🚀 RUN THE AGENT
# ============================================================================

def run_query(query: str, verbose: bool = True):
    """Execute a single query through the agent."""
    agent = create_agent()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🤔 USER: {query}")
        print(f"{'='*70}")
    
    # Create initial state with user message
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Execute the graph
    if verbose:
        print("\n🔄 Agent Execution Steps:")
        step_num = 1
        for step in agent.stream(initial_state):
            node_name = list(step.keys())[0]
            print(f"\n  Step {step_num}: {node_name.upper()}")
            
            last_msg = step[node_name]['messages'][-1]
            
            # Show tool calls if any
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"    🔧 Calling tool: {tc['name']}")
                    print(f"       Args: {tc['args']}")
            
            # Show tool results
            if hasattr(last_msg, 'content') and last_msg.content and node_name == "tools":
                print(f"    📥 Tool result: {last_msg.content[:80]}...")
            
            step_num += 1
    
    # Get final result
    final_state = agent.invoke(initial_state)
    final_message = final_state["messages"][-1]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🤖 AGENT: {final_message.content}")
        print(f"{'='*70}\n")
    
    return final_message.content

# ============================================================================
# 🧪 TEST CASES
# ============================================================================

def run_tests():
    """Run a series of test queries to demonstrate agent capabilities."""
    
    print("\n" + "="*70)
    print("🎯 LANGGRAPH REACT AGENT - TEST SUITE")
    print("="*70)
    
    test_cases = [
        ("What is 157 multiplied by 892?", "Tests calculator tool"),
        ("Tell me about LangGraph", "Tests search tool"),
        ("What's the status of order 12345?", "Tests order lookup tool"),
        ("Look up order 67890", "Tests order lookup with different ID"),
        ("Calculate 2048 divided by 16, then multiply by 3", "Tests multi-step math"),
    ]
    
    for i, (query, description) in enumerate(test_cases, 1):
        print(f"\n\n{'#'*70}")
        print(f"TEST {i}/5: {description}")
        print(f"{'#'*70}")
        
        try:
            run_query(query, verbose=True)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if i < len(test_cases):
            input("\n⏸️  Press Enter to continue to next test...")
    
    print("\n\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    print("\n💡 Next steps:")
    print("  1. Check how the agent decides when to use tools")
    print("  2. Try your own queries!")
    print("  3. Enable LangSmith to see full traces")
    print("  4. Move to Day 1 Afternoon: Add conversation memory!\n")

# ============================================================================
# 🎮 INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Run agent in interactive mode - chat with it!"""
    print("\n" + "="*70)
    print("💬 INTERACTIVE MODE - Chat with the Agent")
    print("="*70)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Available capabilities:")
    print("  - Math calculations")
    print("  - Information search")
    print("  - Order status lookup (try: 12345, 67890, 11111)")
    print("="*70 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if not user_input:
                continue
            
            result = run_query(user_input, verbose=False)
            print(f"\nAgent: {result}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

# ============================================================================
# 🎯 MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if Azure OpenAI is configured
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("\n" + "="*70)
        print("⚠️  SETUP REQUIRED - Azure OpenAI Configuration")
        print("="*70)
        print("\n📝 Missing environment variables in .env file:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n🔧 Please configure these in your .env file:")
        print("   AZURE_OPENAI_API_KEY=your-key-here")
        print("   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("   AZURE_OPENAI_API_VERSION=2024-12-01-preview")
        print("   AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name")
        print("\n" + "="*70 + "\n")
        sys.exit(1)
    
    # Run mode selection
    print("\n" + "="*70)
    print("🚀 LANGGRAPH REACT AGENT - Azure OpenAI Edition")
    print("="*70)
    print(f"📍 Using Azure deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    print(f"📊 LangSmith tracing: {'✓ Enabled' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else '✗ Disabled'}")
    print("="*70)
    print("\nChoose mode:")
    print("  1. Run test suite (recommended for first run)")
    print("  2. Interactive chat mode")
    print("  3. Single query")
    print()
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        run_tests()
    elif choice == "2":
        interactive_mode()
    elif choice == "3":
        query = input("\nEnter your query: ").strip()
        if query:
            run_query(query)
    else:
        print("Invalid choice. Running test suite...")
        run_tests()
