"""
LangGraph Stateful Customer Support Agent - Day 1 Afternoon
Multi-turn conversations with memory, context, and state tracking

New concepts:
- Checkpointing (conversation persistence)
- Rich state schema (not just messages)
- State inspection
- Conversation resumption

Installation:
pip install langgraph langchain-openai langchain-core
"""

import os
from typing import TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
import operator
import json
from datetime import datetime

# ============================================================================
# 🔑 LOAD ENVIRONMENT VARIABLES
# ============================================================================
load_dotenv()

# ============================================================================
# 🛠️ TOOLS (Enhanced with context awareness)
# ============================================================================

@tool
def lookup_customer_info(customer_id: str) -> str:
    """Look up customer information by customer ID."""
    customers = {
        "CUST001": {
            "name": "Alice Johnson",
            "tier": "Premium",
            "account_since": "2023-01-15",
            "total_orders": 47,
            "lifetime_value": "$4,832"
        },
        "CUST002": {
            "name": "Bob Smith",
            "tier": "Standard",
            "account_since": "2024-06-20",
            "total_orders": 5,
            "lifetime_value": "$342"
        },
        "CUST003": {
            "name": "Carol Davis",
            "tier": "Premium",
            "account_since": "2022-03-10",
            "total_orders": 89,
            "lifetime_value": "$12,450"
        }
    }
    
    if customer_id in customers:
        cust = customers[customer_id]
        return (f"👤 Customer: {cust['name']}\n"
                f"   Tier: {cust['tier']}\n"
                f"   Member Since: {cust['account_since']}\n"
                f"   Total Orders: {cust['total_orders']}\n"
                f"   Lifetime Value: {cust['lifetime_value']}")
    
    return f"❌ Customer {customer_id} not found"

@tool
def lookup_order(order_id: str) -> str:
    """Look up order details by order ID."""
    orders = {
        "ORD12345": {
            "customer_id": "CUST001",
            "status": "Shipped",
            "date": "2025-10-10",
            "items": ["Laptop Stand", "USB-C Cable"],
            "total": "$89.99",
            "tracking": "TRK-8372-XYZ",
            "delivery_date": "2025-10-18"
        },
        "ORD67890": {
            "customer_id": "CUST002",
            "status": "Processing",
            "date": "2025-10-15",
            "items": ["Wireless Mouse"],
            "total": "$34.99",
            "tracking": "Pending",
            "delivery_date": "Est. 2025-10-20"
        },
        "ORD11111": {
            "customer_id": "CUST003",
            "status": "Delivered",
            "date": "2025-10-08",
            "items": ["Mechanical Keyboard", "Mouse Pad", "Wrist Rest"],
            "total": "$234.99",
            "tracking": "Delivered",
            "delivery_date": "2025-10-12"
        }
    }
    
    if order_id in orders:
        order = orders[order_id]
        return (f"📦 Order {order_id}:\n"
                f"   Status: {order['status']}\n"
                f"   Date: {order['date']}\n"
                f"   Items: {', '.join(order['items'])}\n"
                f"   Total: {order['total']}\n"
                f"   Tracking: {order['tracking']}\n"
                f"   Delivery: {order['delivery_date']}")
    
    return f"❌ Order {order_id} not found"

@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check if an order is eligible for refund."""
    # Simulated business logic
    orders = {
        "ORD12345": {"eligible": True, "reason": "Within 30-day return window"},
        "ORD67890": {"eligible": True, "reason": "Within 30-day return window"},
        "ORD11111": {"eligible": False, "reason": "Delivered more than 30 days ago"}
    }
    
    if order_id in orders:
        result = orders[order_id]
        status = "✅ Eligible" if result["eligible"] else "❌ Not Eligible"
        return f"{status} for refund\nReason: {result['reason']}"
    
    return f"❌ Cannot check eligibility - order {order_id} not found"

@tool
def process_refund(order_id: str, reason: str) -> str:
    """
    Process a refund for an order. 
    This is a sensitive action that should require human approval.
    """
    # Simulated refund processing
    return (f"💰 Refund initiated for {order_id}\n"
            f"   Reason: {reason}\n"
            f"   Amount will be credited in 3-5 business days\n"
            f"   Reference: REF-{order_id[-4:]}-2025")

@tool
def search_knowledge_base(query: str) -> str:
    """Search company knowledge base for policies and procedures."""
    kb = {
        "refund": "Refund Policy: Items can be returned within 30 days of delivery. Refunds processed in 3-5 business days.",
        "shipping": "Shipping: Standard (5-7 days, free over $50), Express (2-3 days, $15), Overnight ($30)",
        "warranty": "Warranty: All products have 1-year manufacturer warranty. Premium members get extended 2-year warranty.",
        "account": "Account: Update info in profile settings. Contact support for tier upgrades.",
        "password": "Password Reset: Use 'Forgot Password' link on login page or contact support."
    }
    
    query_lower = query.lower()
    for key, value in kb.items():
        if key in query_lower:
            return f"📚 Knowledge Base: {value}"
    
    return "📚 General support: Check our Help Center at support.example.com or contact us"

# ============================================================================
# 📊 ENHANCED STATE SCHEMA
# ============================================================================

class CustomerSupportState(TypedDict):
    """
    Rich state schema for customer support conversations.
    This is the KEY difference from morning session!
    """
    # Conversation history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Customer context
    customer_id: str
    customer_name: str
    customer_tier: str
    
    # Issue tracking
    issue_type: Literal["order_inquiry", "refund_request", "technical_support", "general", "unknown"]
    issue_description: str
    related_order_id: str
    
    # Resolution tracking
    resolution_status: Literal["in_progress", "resolved", "escalated", "needs_approval"]
    resolution_notes: str
    
    # Metadata
    conversation_start: str
    last_updated: str

# ============================================================================
# 🤖 AGENT NODES (Context-Aware)
# ============================================================================

# Initialize LLM with tools
tools = [
    lookup_customer_info,
    lookup_order,
    check_refund_eligibility,
    process_refund,
    search_knowledge_base
]
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    # temperature=0,  # Some Azure models don't support temperature parameter
)
llm_with_tools = llm.bind_tools(tools)

def classify_issue(state: CustomerSupportState):
    """
    Classify the customer's issue based on conversation.
    Updates issue_type in state.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Simple keyword-based classification (in production, use LLM)
    issue_type = "unknown"
    if any(word in last_message.lower() for word in ["refund", "return", "money back"]):
        issue_type = "refund_request"
    elif any(word in last_message.lower() for word in ["order", "track", "delivery", "shipped"]):
        issue_type = "order_inquiry"
    elif any(word in last_message.lower() for word in ["password", "login", "account", "reset"]):
        issue_type = "technical_support"
    else:
        issue_type = "general"
    
    return {
        "issue_type": issue_type,
        "last_updated": datetime.now().isoformat()
    }

def agent_node(state: CustomerSupportState):
    """
    Main agent node with context awareness.
    """
    messages = state["messages"]
    
    # Build context-aware system message
    system_context = f"""You are a helpful customer support agent.

CURRENT CONTEXT:
- Customer: {state.get('customer_name', 'Unknown')} (ID: {state.get('customer_id', 'Unknown')})
- Tier: {state.get('customer_tier', 'Unknown')}
- Issue Type: {state.get('issue_type', 'Unknown')}
- Status: {state.get('resolution_status', 'in_progress')}

GUIDELINES:
- Be empathetic and professional
- Use customer's name when appropriate
- Premium customers get priority treatment
- For refunds, check eligibility first, then process
- Always provide tracking info for orders
- Summarize resolution at end of conversation
"""
    
    # Add system message if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_context)] + list(messages)
    
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "last_updated": datetime.now().isoformat()
    }

def should_continue(state: CustomerSupportState):
    """Routing logic with state awareness."""
    last_message = state["messages"][-1]
    
    # Check if we need human approval for sensitive actions
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "process_refund":
                return "needs_approval"
    
    # Normal tool routing
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"

def approval_gate(state: CustomerSupportState):
    """
    Human-in-the-loop approval node for sensitive actions.
    In production, this would pause and wait for human approval.
    """
    last_message = state["messages"][-1]
    
    print("\n" + "="*70)
    print("⚠️  APPROVAL REQUIRED - Sensitive Action Detected")
    print("="*70)
    
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "process_refund":
            print(f"\n🔍 Action: Process Refund")
            print(f"   Order ID: {tool_call['args'].get('order_id')}")
            print(f"   Reason: {tool_call['args'].get('reason')}")
            print(f"   Customer: {state.get('customer_name')} ({state.get('customer_tier')})")
    
    approval = input("\n✋ Approve this action? (yes/no): ").strip().lower()
    
    if approval == "yes":
        print("✅ Approved - Processing...")
        return {
            "resolution_status": "resolved",
            "resolution_notes": "Refund approved and processed"
        }
    else:
        print("❌ Denied - Action cancelled")
        # Remove tool calls to prevent execution
        last_message.tool_calls = []
        return {
            "messages": [AIMessage(content="I apologize, but I cannot process this refund at this time. A supervisor will review your case.")],
            "resolution_status": "escalated",
            "resolution_notes": "Refund denied by approval gate"
        }

# ============================================================================
# 🗺️ BUILD STATEFUL GRAPH WITH CHECKPOINTING
# ============================================================================

def create_stateful_agent():
    """
    Create agent with memory and checkpointing.
    This allows conversations to be saved and resumed!
    """
    # Memory saver for checkpointing
    memory = MemorySaver()
    
    workflow = StateGraph(CustomerSupportState)
    
    # Add nodes
    workflow.add_node("classify", classify_issue)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("approval_gate", approval_gate)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Flow: classify → agent → (tools/approval/end)
    workflow.add_edge("classify", "agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "needs_approval": "approval_gate",
            "end": END
        }
    )
    
    # After approval, go to tools
    workflow.add_edge("approval_gate", "tools")
    
    # After tools, back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with checkpointing
    return workflow.compile(checkpointer=memory)

# ============================================================================
# 🚀 CONVERSATION MANAGER
# ============================================================================

class ConversationSession:
    """Manages a stateful conversation session."""
    
    def __init__(self, customer_id: str, customer_name: str, customer_tier: str):
        self.agent = create_stateful_agent()
        self.session_id = f"session_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize state
        self.config = {"configurable": {"thread_id": self.session_id}}
        self.initial_state = {
            "messages": [],
            "customer_id": customer_id,
            "customer_name": customer_name,
            "customer_tier": customer_tier,
            "issue_type": "unknown",
            "issue_description": "",
            "related_order_id": "",
            "resolution_status": "in_progress",
            "resolution_notes": "",
            "conversation_start": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def send_message(self, message: str):
        """Send a message and get response."""
        # Add user message to state
        current_state = self.agent.get_state(self.config)
        
        if current_state.values:
            # Continue existing conversation
            input_state = {"messages": [HumanMessage(content=message)]}
        else:
            # Start new conversation
            input_state = {
                **self.initial_state,
                "messages": [HumanMessage(content=message)]
            }
        
        # Execute graph
        result = self.agent.invoke(input_state, self.config)
        
        # Return last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content, result
        
        return "No response generated", result
    
    def get_state(self):
        """Get current conversation state."""
        return self.agent.get_state(self.config)
    
    def print_state_summary(self):
        """Print summary of conversation state."""
        state = self.get_state()
        if state.values:
            print(f"\n{'='*70}")
            print("📊 CONVERSATION STATE")
            print(f"{'='*70}")
            print(f"Customer: {state.values.get('customer_name')} ({state.values.get('customer_tier')})")
            print(f"Issue Type: {state.values.get('issue_type')}")
            print(f"Status: {state.values.get('resolution_status')}")
            print(f"Messages: {len(state.values.get('messages', []))}")
            print(f"Last Updated: {state.values.get('last_updated')}")
            print(f"{'='*70}\n")

# ============================================================================
# 🧪 DEMO SCENARIOS
# ============================================================================

def demo_multi_turn_conversation():
    """Demo: Multi-turn conversation with memory."""
    print("\n" + "="*70)
    print("🎬 DEMO 1: Multi-Turn Conversation with Memory")
    print("="*70)
    
    # Start conversation
    session = ConversationSession(
        customer_id="CUST001",
        customer_name="Alice Johnson",
        customer_tier="Premium"
    )
    
    print("\n👤 Customer: Alice Johnson (Premium)")
    print("="*70)
    
    # Turn 1
    print("\n💬 Turn 1:")
    print("User: Hi, I'd like to check on my recent order")
    response, _ = session.send_message("Hi, I'd like to check on my recent order")
    print(f"Agent: {response}")
    
    input("\n⏸️  Press Enter to continue...")
    
    # Turn 2 - Agent should remember context
    print("\n💬 Turn 2:")
    print("User: It's order ORD12345")
    response, _ = session.send_message("It's order ORD12345")
    print(f"Agent: {response}")
    
    input("\n⏸️  Press Enter to continue...")
    
    # Turn 3 - Still maintaining context
    print("\n💬 Turn 3:")
    print("User: Actually, I'd like to return it")
    response, _ = session.send_message("Actually, I'd like to return it")
    print(f"Agent: {response}")
    
    # Show final state
    session.print_state_summary()

def demo_conversation_resumption():
    """Demo: Save and resume conversation."""
    print("\n\n" + "="*70)
    print("🎬 DEMO 2: Conversation Interruption & Resumption")
    print("="*70)
    
    session = ConversationSession(
        customer_id="CUST002",
        customer_name="Bob Smith",
        customer_tier="Standard"
    )
    
    print("\n👤 Customer: Bob Smith (Standard)")
    print("="*70)
    
    # Start conversation
    print("\n💬 Starting conversation...")
    response, _ = session.send_message("I want to check my order ORD67890")
    print(f"Agent: {response[:100]}...")
    
    print("\n⏸️  << CONVERSATION INTERRUPTED >>")
    print("(Simulating customer closing chat window)")
    
    input("\nPress Enter to resume conversation...")
    
    print("\n▶️  << CONVERSATION RESUMED >>")
    print("Agent has full context from previous session!")
    
    # Continue conversation - should remember everything
    response, _ = session.send_message("When will it arrive?")
    print(f"\n💬 User: When will it arrive?")
    print(f"Agent: {response}")
    
    session.print_state_summary()

def interactive_mode():
    """Interactive customer support simulation."""
    print("\n" + "="*70)
    print("💬 INTERACTIVE CUSTOMER SUPPORT")
    print("="*70)
    
    # Get customer info
    print("\nAvailable customers:")
    print("  1. Alice Johnson (CUST001) - Premium")
    print("  2. Bob Smith (CUST002) - Standard")
    print("  3. Carol Davis (CUST003) - Premium")
    
    choice = input("\nSelect customer (1-3): ").strip()
    
    customers = {
        "1": ("CUST001", "Alice Johnson", "Premium"),
        "2": ("CUST002", "Bob Smith", "Standard"),
        "3": ("CUST003", "Carol Davis", "Premium")
    }
    
    if choice not in customers:
        print("Invalid choice")
        return
    
    cust_id, cust_name, cust_tier = customers[choice]
    session = ConversationSession(cust_id, cust_name, cust_tier)
    
    print(f"\n{'='*70}")
    print(f"Conversation started with {cust_name} ({cust_tier})")
    print(f"{'='*70}")
    print("Type 'state' to see conversation state")
    print("Type 'quit' to end conversation")
    print(f"{'='*70}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                session.print_state_summary()
                print("\n👋 Conversation ended\n")
                break
            
            if user_input.lower() == 'state':
                session.print_state_summary()
                continue
            
            if not user_input:
                continue
            
            response, _ = session.send_message(user_input)
            print(f"\nAgent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Conversation ended\n")
            break

# ============================================================================
# 🎯 MAIN
# ============================================================================

if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY") == "your-key-here":
        print("\n⚠️  Please set your OpenAI API key in the code!\n")
        exit(1)
    
    print("\n" + "="*70)
    print("🚀 STATEFUL CUSTOMER SUPPORT AGENT - DAY 1 AFTERNOON")
    print("="*70)
    print("\nChoose mode:")
    print("  1. Demo: Multi-turn conversation with memory")
    print("  2. Demo: Conversation interruption & resumption")
    print("  3. Interactive mode (chat as customer)")
    print()
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        demo_multi_turn_conversation()
    elif choice == "2":
        demo_conversation_resumption()
    elif choice == "3":
        interactive_mode()
    else:
        print("Running all demos...")
        demo_multi_turn_conversation()
        input("\n\nPress Enter for next demo...")
        demo_conversation_resumption()
