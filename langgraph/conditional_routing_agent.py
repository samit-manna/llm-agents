"""
Day 2 Morning: Smart Conditional Routing Agent
Multi-path routing based on issue classification with specialized handlers
Modified to work with Azure OpenAI

New Concepts:
- Conditional edges based on state
- Multiple specialized handler nodes
- Dynamic routing logic
- Issue classification
- Handler specialization

"""

import os
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import operator
from datetime import datetime

# ============================================================================
# 🔑 SETUP
# ============================================================================
load_dotenv()

# Azure OpenAI configuration from .env:
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_VERSION
# - AZURE_OPENAI_DEPLOYMENT_NAME

# ============================================================================
# 🛠️ SPECIALIZED TOOLS BY DOMAIN
# ============================================================================

# === BILLING TOOLS ===
@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up invoice details."""
    invoices = {
        "INV-001": "Invoice $89.99 - Paid on Oct 1, 2025",
        "INV-002": "Invoice $234.50 - Payment pending",
        "INV-003": "Invoice $45.00 - Overdue since Sep 15"
    }
    return invoices.get(invoice_id, "Invoice not found")

@tool
def check_payment_status(customer_id: str) -> str:
    """Check customer payment status."""
    payments = {
        "CUST001": "All payments current. Next bill: Nov 1",
        "CUST002": "Payment overdue: $45.00 from Sep 15",
        "CUST003": "Auto-pay enabled. No action needed"
    }
    return payments.get(customer_id, "Customer not found")

@tool
def process_refund(order_id: str, amount: float, reason: str) -> str:
    """Process a refund."""
    return f"✅ Refund of ${amount:.2f} initiated for {order_id}. Reason: {reason}. ETA: 3-5 business days"

# === TECHNICAL SUPPORT TOOLS ===
@tool
def check_system_status(service: str) -> str:
    """Check if a system/service is operational."""
    status = {
        "api": "✅ Operational - 99.9% uptime",
        "website": "✅ Operational",
        "mobile_app": "⚠️ Degraded performance - investigating",
        "payment_gateway": "✅ Operational"
    }
    return status.get(service.lower(), "Service not found")

@tool
def reset_password(email: str) -> str:
    """Send password reset link."""
    return f"✅ Password reset link sent to {email}. Check spam folder if not received in 5 minutes."

@tool
def search_knowledge_base(query: str) -> str:
    """Search technical documentation."""
    kb = {
        "api": "API Documentation: Use Bearer token authentication. Rate limit: 1000 req/hour",
        "integration": "Integration Guide: Follow OAuth 2.0 flow. SDK available for Python, Node.js",
        "error": "Common errors: 401 (auth failed), 429 (rate limit), 500 (server error)",
        "ssl": "SSL Certificate: Valid until Jan 2026. Auto-renewal enabled"
    }
    for key, value in kb.items():
        if key in query.lower():
            return f"📚 {value}"
    return "📚 No specific documentation found. Contact support@example.com"

# === ORDER MANAGEMENT TOOLS ===
@tool
def lookup_order(order_id: str) -> str:
    """Look up order details."""
    orders = {
        "ORD12345": "Status: Shipped (Oct 10) → Delivery: Oct 18. Items: Laptop Stand, USB-C Cable",
        "ORD67890": "Status: Processing → Est. Ship: Oct 17. Items: Wireless Mouse",
        "ORD11111": "Status: Delivered (Oct 12). Items: Keyboard, Mouse Pad, Wrist Rest"
    }
    return orders.get(order_id, "Order not found")

@tool
def track_shipment(tracking_number: str) -> str:
    """Track package shipment."""
    tracking = {
        "TRK001": "📦 In transit → Last scan: Chicago, IL (Oct 15, 3:42 PM) → Next: Your city",
        "TRK002": "📦 Out for delivery → Expected today by 8 PM",
        "TRK003": "✅ Delivered (Oct 12, 2:30 PM) → Signed by: J. Smith"
    }
    return tracking.get(tracking_number, "Tracking number not found")

@tool
def modify_order(order_id: str, modification: str) -> str:
    """Modify an order (address, items, etc)."""
    return f"✅ Modification requested for {order_id}: {modification}. Processing within 2 hours."

# ============================================================================
# 📊 ENHANCED STATE WITH ROUTING
# ============================================================================

class RouterState(TypedDict):
    """State with routing information."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Customer info
    customer_id: str
    customer_name: str
    customer_tier: str
    
    # Classification
    issue_type: Literal[
        "billing",
        "technical",
        "order_management",
        "general",
        "unknown"
    ]
    confidence: float  # 0-1 confidence in classification
    
    # Routing
    current_handler: str
    requires_escalation: bool
    escalation_reason: str
    
    # Resolution
    resolution_status: Literal["in_progress", "resolved", "escalated"]
    attempts: int  # Number of handler attempts

# ============================================================================
# 🎯 CLASSIFIER NODE
# ============================================================================

def classify_issue_node(state: RouterState):
    """
    Intelligent issue classification using LLM.
    Routes to appropriate specialized handler.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Use Azure OpenAI for classification
    classifier_llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    
    classification_prompt = f"""Classify this customer support query into ONE category:

Query: {last_message}

Categories:
1. billing - invoices, payments, refunds, charges
2. technical - system issues, errors, API, integration, password reset
3. order_management - order status, tracking, delivery, modifications
4. general - questions, information, policies

Respond with ONLY the category name and confidence (0-1), like:
billing 0.95
"""
    
    response = classifier_llm.invoke([HumanMessage(content=classification_prompt)])
    
    # Parse response
    try:
        parts = response.content.strip().split()
        issue_type = parts[0]
        confidence = float(parts[1]) if len(parts) > 1 else 0.8
    except:
        issue_type = "general"
        confidence = 0.5
    
    print(f"\n🎯 CLASSIFIER: {issue_type} (confidence: {confidence:.2f})")
    
    return {
        "issue_type": issue_type,
        "confidence": confidence,
        "attempts": 0
    }

# ============================================================================
# 🔧 SPECIALIZED HANDLER NODES
# ============================================================================

# Billing handler tools
billing_tools = [lookup_invoice, check_payment_status, process_refund]
billing_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
).bind_tools(billing_tools)

def billing_handler_node(state: RouterState):
    """Specialized handler for billing issues."""
    messages = state["messages"]
    
    system_msg = SystemMessage(content="""You are a billing specialist.
Handle: invoices, payments, refunds, charges.
Tools: lookup_invoice, check_payment_status, process_refund.
Be professional and clear about financial matters.""")
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_msg] + list(messages)
    
    print("💰 BILLING HANDLER activated")
    response = billing_llm.invoke(messages)
    
    return {
        "messages": [response],
        "current_handler": "billing",
        "attempts": state.get("attempts", 0) + 1
    }

# Technical support handler tools
tech_tools = [check_system_status, reset_password, search_knowledge_base]
tech_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
).bind_tools(tech_tools)

def tech_handler_node(state: RouterState):
    """Specialized handler for technical issues."""
    messages = state["messages"]
    
    system_msg = SystemMessage(content="""You are a technical support specialist.
Handle: system errors, API issues, integrations, password resets.
Tools: check_system_status, reset_password, search_knowledge_base.
Provide clear technical solutions.""")
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_msg] + list(messages)
    
    print("🔧 TECH SUPPORT HANDLER activated")
    response = tech_llm.invoke(messages)
    
    return {
        "messages": [response],
        "current_handler": "technical",
        "attempts": state.get("attempts", 0) + 1
    }

# Order management handler tools
order_tools = [lookup_order, track_shipment, modify_order]
order_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
).bind_tools(order_tools)

def order_handler_node(state: RouterState):
    """Specialized handler for order management."""
    messages = state["messages"]
    
    system_msg = SystemMessage(content="""You are an order management specialist.
Handle: order status, tracking, delivery, modifications.
Tools: lookup_order, track_shipment, modify_order.
Focus on quick resolution of order issues.""")
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_msg] + list(messages)
    
    print("📦 ORDER HANDLER activated")
    response = order_llm.invoke(messages)
    
    return {
        "messages": [response],
        "current_handler": "order_management",
        "attempts": state.get("attempts", 0) + 1
    }

# General handler (no specialized tools)
general_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

def general_handler_node(state: RouterState):
    """General handler for non-specialized queries."""
    messages = state["messages"]
    
    system_msg = SystemMessage(content="""You are a general customer support agent.
Handle: general questions, information requests, policies.
Provide helpful information and direct to specialists if needed.""")
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_msg] + list(messages)
    
    print("ℹ️ GENERAL HANDLER activated")
    response = general_llm.invoke(messages)
    
    return {
        "messages": [response],
        "current_handler": "general",
        "attempts": state.get("attempts", 0) + 1
    }

# ============================================================================
# 🔀 ROUTING LOGIC
# ============================================================================

def route_to_handler(state: RouterState) -> str:
    """
    Route to appropriate handler based on classification.
    This is the KEY conditional edge!
    """
    issue_type = state.get("issue_type", "unknown")
    confidence = state.get("confidence", 0)
    
    # Low confidence → go to general handler for clarification
    if confidence < 0.6:
        print(f"⚠️ Low confidence ({confidence:.2f}) → routing to general handler")
        return "general_handler"
    
    # Route based on issue type
    routing_map = {
        "billing": "billing_handler",
        "technical": "tech_handler",
        "order_management": "order_handler",
        "general": "general_handler"
    }
    
    handler = routing_map.get(issue_type, "general_handler")
    print(f"→ Routing to: {handler}")
    
    return handler

def should_use_tools(state: RouterState) -> str:
    """Check if handler wants to use tools."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"

def route_back_to_handler(state: RouterState) -> str:
    """
    Route back to the handler that called the tools.
    Uses the current_handler from state.
    """
    current_handler = state.get("current_handler", "general")
    
    # Map handler names to node names
    handler_map = {
        "billing": "billing_handler",
        "technical": "tech_handler",
        "order_management": "order_handler",
        "general": "general_handler"
    }
    
    handler_node = handler_map.get(current_handler, "general_handler")
    print(f"  ↩️  Returning to: {handler_node}")
    
    return handler_node

# ============================================================================
# 🗺️ BUILD ROUTING GRAPH
# ============================================================================

def create_routing_agent():
    """
    Create agent with conditional routing to specialized handlers.
    
    Flow:
    START → classify → route_to_handler → [billing/tech/order/general] 
          → tools (if needed) → back to handler → END
    """
    memory = MemorySaver()
    workflow = StateGraph(RouterState)
    
    # Add all nodes
    workflow.add_node("classify", classify_issue_node)
    workflow.add_node("billing_handler", billing_handler_node)
    workflow.add_node("tech_handler", tech_handler_node)
    workflow.add_node("order_handler", order_handler_node)
    workflow.add_node("general_handler", general_handler_node)
    
    # Tool nodes for each domain
    workflow.add_node("tools", ToolNode(
        billing_tools + tech_tools + order_tools
    ))
    
    # Entry point
    workflow.set_entry_point("classify")
    
    # Conditional routing from classifier
    workflow.add_conditional_edges(
        "classify",
        route_to_handler,
        {
            "billing_handler": "billing_handler",
            "tech_handler": "tech_handler",
            "order_handler": "order_handler",
            "general_handler": "general_handler"
        }
    )
    
    # Each handler can use tools or end
    for handler in ["billing_handler", "tech_handler", "order_handler", "general_handler"]:
        workflow.add_conditional_edges(
            handler,
            should_use_tools,
            {
                "tools": "tools",
                "end": END
            }
        )
    
    # After tools, route back to the correct handler based on state
    workflow.add_conditional_edges(
        "tools",
        route_back_to_handler,
        {
            "billing_handler": "billing_handler",
            "tech_handler": "tech_handler",
            "order_handler": "order_handler",
            "general_handler": "general_handler"
        }
    )
    
    return workflow.compile(checkpointer=memory)

# ============================================================================
# 🧪 TEST SCENARIOS
# ============================================================================

def test_routing():
    """Test different routing scenarios."""
    agent = create_routing_agent()
    
    test_cases = [
        ("Check my invoice INV-001", "billing"),
        ("My password doesn't work", "technical"),
        ("Where is order ORD12345?", "order_management"),
        ("What are your business hours?", "general"),
        ("I was charged twice for INV-002", "billing"),
        ("The API returns 401 error", "technical"),
        ("Track package TRK001", "order_management"),
    ]
    
    print("\n" + "="*70)
    print("🧪 CONDITIONAL ROUTING TEST SUITE")
    print("="*70)
    
    for i, (query, expected_route) in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(test_cases)}: {query}")
        print(f"Expected route: {expected_route}")
        print(f"{'─'*70}")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "customer_id": "CUST001",
            "customer_name": "Alice",
            "customer_tier": "Premium",
            "issue_type": "unknown",
            "confidence": 0.0,
            "current_handler": "",
            "requires_escalation": False,
            "escalation_reason": "",
            "resolution_status": "in_progress",
            "attempts": 0
        }
        
        config = {"configurable": {"thread_id": f"test_{i}"}}
        
        try:
            result = agent.invoke(initial_state, config)
            
            # Get final response
            final_msg = result["messages"][-1]
            actual_route = result.get("current_handler", "unknown")
            
            print(f"\n✅ Routed to: {actual_route}")
            print(f"📝 Response: {final_msg.content[:100]}...")
            
            if expected_route in actual_route:
                print("✅ CORRECT ROUTING")
            else:
                print(f"⚠️ UNEXPECTED ROUTING (expected: {expected_route})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if i < len(test_cases):
            input("\n⏸️  Press Enter for next test...")

def interactive_routing():
    """Interactive mode to test routing."""
    agent = create_routing_agent()
    
    print("\n" + "="*70)
    print("💬 INTERACTIVE ROUTING MODE")
    print("="*70)
    print("\nThe agent will classify your query and route to specialist.")
    print("Watch which handler gets activated!")
    print("Type 'quit' to exit\n")
    print("="*70)
    
    session_id = "interactive_session"
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if not query:
                continue
            
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "customer_id": "CUST001",
                "customer_name": "Alice",
                "customer_tier": "Premium",
                "issue_type": "unknown",
                "confidence": 0.0,
                "current_handler": "",
                "requires_escalation": False,
                "escalation_reason": "",
                "resolution_status": "in_progress",
                "attempts": 0
            }
            
            config = {"configurable": {"thread_id": session_id}}
            result = agent.invoke(initial_state, config)
            
            # Show routing info
            print(f"\n📊 Classification: {result['issue_type']} (confidence: {result['confidence']:.2f})")
            print(f"🎯 Handler: {result['current_handler']}")
            
            # Show response
            final_msg = result["messages"][-1]
            print(f"\nAgent: {final_msg.content}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

# ============================================================================
# 🎯 MAIN
# ============================================================================

if __name__ == "__main__":
    # Check Azure OpenAI configuration
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
        print("\n🔧 Please configure these in your .env file")
        print("="*70 + "\n")
        exit(1)
    
    print("\n" + "="*70)
    print("🚀 DAY 2 MORNING - CONDITIONAL ROUTING AGENT")
    print("="*70)
    print(f"📍 Using Azure deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    print("="*70)
    print("\nFeatures:")
    print("  ✅ Intelligent issue classification")
    print("  ✅ Specialized handlers (billing, tech, orders)")
    print("  ✅ Conditional routing based on issue type")
    print("  ✅ Domain-specific tools per handler")
    print("\nChoose mode:")
    print("  1. Run test suite (see routing in action)")
    print("  2. Interactive mode (test your queries)")
    print()
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "1":
        test_routing()
    elif choice == "2":
        interactive_routing()
    else:
        print("Running test suite...")
        test_routing()
