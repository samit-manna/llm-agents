"""
Day 2 Afternoon: Production Error Handling & Resilience
Retry logic, fallback paths, circuit breakers, max iterations
Modified to work with Azure OpenAI

New Concepts:
- Error handling nodes
- Retry logic with exponential backoff
- Fallback paths when tools fail
- Circuit breakers for unreliable services
- Max iteration limits (prevent infinite loops)
- Graceful degradation

Installation:
pip install langgraph langchain-openai langchain-core tenacity python-dotenv
"""

import os
from typing import TypedDict, Annotated, Sequence, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import operator
import time
import random
from datetime import datetime, timedelta
from collections import defaultdict

# For retry logic
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    print("⚠️  Install tenacity for advanced retry: pip install tenacity")

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
# 🔧 CIRCUIT BREAKER IMPLEMENTATION
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for tool failures.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # seconds before trying again
        self.failures = defaultdict(int)
        self.last_failure_time = defaultdict(float)
        self.state = defaultdict(lambda: "CLOSED")  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, tool_name: str, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        # Check circuit state
        if self.state[tool_name] == "OPEN":
            # Check if timeout has passed
            if time.time() - self.last_failure_time[tool_name] > self.timeout:
                print(f"🔄 Circuit HALF_OPEN for {tool_name} - attempting recovery")
                self.state[tool_name] = "HALF_OPEN"
            else:
                remaining = int(self.timeout - (time.time() - self.last_failure_time[tool_name]))
                raise Exception(f"🚫 Circuit OPEN for {tool_name} - retry in {remaining}s")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failures
            if self.state[tool_name] == "HALF_OPEN":
                print(f"✅ Circuit CLOSED for {tool_name} - service recovered")
                self.state[tool_name] = "CLOSED"
                self.failures[tool_name] = 0
            
            return result
            
        except Exception as e:
            # Failure - increment counter
            self.failures[tool_name] += 1
            self.last_failure_time[tool_name] = time.time()
            
            print(f"❌ Tool failure #{self.failures[tool_name]} for {tool_name}")
            
            # Open circuit if threshold reached
            if self.failures[tool_name] >= self.failure_threshold:
                self.state[tool_name] = "OPEN"
                print(f"🚨 Circuit OPEN for {tool_name} - too many failures!")
            
            raise

# Global circuit breaker
circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)

# ============================================================================
# 🛠️ UNRELIABLE TOOLS (Simulating Real-World Failures)
# ============================================================================

class UnreliableService:
    """Simulates an unreliable external service."""
    
    def __init__(self, failure_rate: float = 0.3):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    def should_fail(self) -> bool:
        """Randomly fail based on failure rate."""
        self.call_count += 1
        return random.random() < self.failure_rate

# Simulate unreliable services
payment_service = UnreliableService(failure_rate=0.4)  # Fails 40% of the time
inventory_service = UnreliableService(failure_rate=0.2)  # Fails 20% of the time
shipping_service = UnreliableService(failure_rate=0.1)  # Fails 10% of the time

@tool
def check_payment_api(customer_id: str) -> str:
    """Check payment status (unreliable external API)."""
    time.sleep(0.1)  # Simulate API call
    
    if payment_service.should_fail():
        raise TimeoutError("Payment API timeout")
    
    return f"Payment status for {customer_id}: All current"

@tool
def check_inventory_api(product_id: str) -> str:
    """Check inventory (sometimes fails)."""
    time.sleep(0.1)
    
    if inventory_service.should_fail():
        raise ConnectionError("Inventory service unavailable")
    
    return f"Product {product_id}: 47 units in stock"

@tool
def check_shipping_api(order_id: str) -> str:
    """Check shipping status (mostly reliable)."""
    time.sleep(0.1)
    
    if shipping_service.should_fail():
        raise Exception("Shipping service error")
    
    return f"Order {order_id}: In transit, arriving Oct 18"

# ============================================================================
# 🔄 TOOLS WITH RETRY LOGIC
# ============================================================================

def retry_with_backoff(func, max_attempts: int = 3):
    """
    Manual retry with exponential backoff.
    Fallback if tenacity not available.
    """
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            print(f"⏳ Retrying in {wait_time}s...")
            time.sleep(wait_time)

@tool
def resilient_check_payment(customer_id: str) -> str:
    """Check payment with retry and circuit breaker."""
    try:
        return circuit_breaker.call(
            "payment_api",
            retry_with_backoff,
            lambda: check_payment_api.invoke({"customer_id": customer_id})
        )
    except Exception as e:
        # Fallback to cached data
        print(f"💾 Using cached payment data (API failed)")
        return f"Payment status for {customer_id}: Unable to verify (showing cached data)"

@tool
def resilient_check_inventory(product_id: str) -> str:
    """Check inventory with retry."""
    try:
        return retry_with_backoff(
            lambda: check_inventory_api.invoke({"product_id": product_id})
        )
    except Exception as e:
        # Fallback to estimate
        print(f"💾 Using estimated inventory")
        return f"Product {product_id}: Limited stock available (estimate)"

# ============================================================================
# 📊 STATE WITH ERROR TRACKING
# ============================================================================

class ResilientState(TypedDict):
    """State with error tracking and retry metadata."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Customer context
    customer_id: str
    issue_type: str
    
    # Error tracking
    errors: list[dict]  # List of errors encountered
    retry_count: int
    max_retries: int
    
    # Iteration control
    iteration_count: int
    max_iterations: int
    
    # Fallback mode
    degraded_mode: bool
    fallback_reason: str

# ============================================================================
# 🤖 AGENT WITH ERROR HANDLING
# ============================================================================

resilient_tools = [
    resilient_check_payment,
    resilient_check_inventory,
    check_shipping_api  # Original tool without retry
]

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
llm_with_tools = llm.bind_tools(resilient_tools)

def agent_node(state: ResilientState):
    """Agent node with error handling."""
    messages = state["messages"]
    iteration = state.get("iteration_count", 0)
    
    # Check max iterations (prevent infinite loops)
    if iteration >= state.get("max_iterations", 10):
        print(f"🚨 MAX ITERATIONS REACHED ({iteration})")
        return {
            "messages": [AIMessage(content="I apologize, but I'm unable to complete this request after multiple attempts. Please contact support.")],
            "resolution_status": "escalated",
            "degraded_mode": True,
            "fallback_reason": "max_iterations_exceeded"
        }
    
    try:
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "iteration_count": iteration + 1
        }
        
    except Exception as e:
        print(f"❌ Agent error: {e}")
        
        # Record error
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "iteration": iteration
        }
        
        errors = state.get("errors", [])
        errors.append(error_record)
        
        return {
            "errors": errors,
            "retry_count": state.get("retry_count", 0) + 1,
            "iteration_count": iteration + 1
        }

def error_handler_node(state: ResilientState):
    """
    Error handling node - decides whether to retry or fallback.
    """
    errors = state.get("errors", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    print(f"\n🔧 ERROR HANDLER - Retry {retry_count}/{max_retries}")
    
    if not errors:
        return {}
    
    last_error = errors[-1]
    print(f"   Last error: {last_error['error_type']}")
    
    # Check if we should retry
    if retry_count < max_retries:
        # Exponential backoff
        wait_time = 2 ** retry_count
        print(f"   ⏳ Waiting {wait_time}s before retry...")
        time.sleep(wait_time)
        
        return {
            "messages": [SystemMessage(content=f"Previous attempt failed. Retrying with alternative approach...")]
        }
    else:
        # Max retries exceeded - enter degraded mode
        print(f"   🚨 Max retries exceeded - entering degraded mode")
        
        return {
            "degraded_mode": True,
            "fallback_reason": f"max_retries_exceeded_after_{retry_count}_attempts",
            "messages": [AIMessage(content="I'm experiencing technical difficulties. Let me provide what information I can from cached data.")]
        }

def should_retry(state: ResilientState) -> str:
    """Decide whether to retry or continue."""
    errors = state.get("errors", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    # If errors and under retry limit
    if errors and retry_count < max_retries:
        return "retry"
    
    # Check for tool calls
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"

# ============================================================================
# 🗺️ BUILD RESILIENT GRAPH
# ============================================================================

def create_resilient_agent():
    """
    Create agent with error handling and retry logic.
    
    Flow:
    START → agent → should_retry?
                        ├→ tools → agent
                        ├→ retry → error_handler → agent
                        └→ end
    """
    memory = MemorySaver()
    workflow = StateGraph(ResilientState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(resilient_tools))
    workflow.add_node("error_handler", error_handler_node)
    
    # Entry point
    workflow.set_entry_point("agent")
    
    # Conditional routing from agent
    workflow.add_conditional_edges(
        "agent",
        should_retry,
        {
            "tools": "tools",
            "retry": "error_handler",
            "end": END
        }
    )
    
    # After tools, back to agent
    workflow.add_edge("tools", "agent")
    
    # After error handling, back to agent
    workflow.add_edge("error_handler", "agent")
    
    return workflow.compile(checkpointer=memory)

# ============================================================================
# 🧪 TEST ERROR SCENARIOS
# ============================================================================

def test_error_handling():
    """Test error handling scenarios."""
    agent = create_resilient_agent()
    
    test_cases = [
        {
            "name": "Payment API (40% failure rate)",
            "query": "Check payment status for CUST001",
            "expected": "Should retry and potentially use cached data"
        },
        {
            "name": "Inventory API (20% failure rate)",
            "query": "Check inventory for PROD123",
            "expected": "May succeed or fallback to estimate"
        },
        {
            "name": "Multiple tools (cascading failures)",
            "query": "Check payment for CUST001 and inventory for PROD456",
            "expected": "Should handle partial failures gracefully"
        }
    ]
    
    print("\n" + "="*70)
    print("🧪 ERROR HANDLING & RESILIENCE TEST SUITE")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(test_cases)}: {test['name']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")
        print(f"{'─'*70}")
        
        initial_state = {
            "messages": [HumanMessage(content=test['query'])],
            "customer_id": "CUST001",
            "issue_type": "billing",
            "errors": [],
            "retry_count": 0,
            "max_retries": 3,
            "iteration_count": 0,
            "max_iterations": 10,
            "degraded_mode": False,
            "fallback_reason": ""
        }
        
        config = {"configurable": {"thread_id": f"test_{i}"}}
        
        try:
            start_time = time.time()
            result = agent.invoke(initial_state, config)
            elapsed = time.time() - start_time
            
            # Show results
            final_msg = result["messages"][-1]
            print(f"\n📊 Results:")
            print(f"   Response: {final_msg.content[:100]}...")
            print(f"   Errors encountered: {len(result.get('errors', []))}")
            print(f"   Retries: {result.get('retry_count', 0)}")
            print(f"   Iterations: {result.get('iteration_count', 0)}")
            print(f"   Degraded mode: {result.get('degraded_mode', False)}")
            print(f"   Time: {elapsed:.2f}s")
            
            if result.get('errors'):
                print(f"\n   Error details:")
                for err in result['errors']:
                    print(f"      - {err['error_type']}: {err['error_message']}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        if i < len(test_cases):
            input("\n⏸️  Press Enter for next test...")
    
    # Show circuit breaker status
    print(f"\n{'='*70}")
    print("🔌 CIRCUIT BREAKER STATUS")
    print(f"{'='*70}")
    for tool_name, state in circuit_breaker.state.items():
        failures = circuit_breaker.failures[tool_name]
        print(f"   {tool_name}: {state} (failures: {failures})")

def demo_max_iterations():
    """Demo: Max iteration protection."""
    print("\n\n" + "="*70)
    print("🔄 DEMO: Max Iteration Protection")
    print("="*70)
    print("\nSimulating an agent that might loop infinitely...")
    
    agent = create_resilient_agent()
    
    initial_state = {
        "messages": [HumanMessage(content="Check payment for CUST001")],
        "customer_id": "CUST001",
        "issue_type": "billing",
        "errors": [],
        "retry_count": 0,
        "max_retries": 2,  # Lower for demo
        "iteration_count": 0,
        "max_iterations": 5,  # Lower for demo
        "degraded_mode": False,
        "fallback_reason": ""
    }
    
    config = {"configurable": {"thread_id": "max_iter_demo"}}
    
    print("\n🔄 Running with max_iterations=5...")
    result = agent.invoke(initial_state, config)
    
    print(f"\n📊 Results:")
    print(f"   Total iterations: {result['iteration_count']}")
    print(f"   Stopped at max: {result['iteration_count'] >= result['max_iterations']}")
    print(f"   Degraded mode: {result.get('degraded_mode', False)}")

def interactive_resilience():
    """Interactive mode to test resilience."""
    agent = create_resilient_agent()
    
    print("\n" + "="*70)
    print("💬 INTERACTIVE RESILIENCE TESTING")
    print("="*70)
    print("\nTools will fail randomly - watch error handling!")
    print("Type 'quit' to exit, 'status' for circuit breaker state\n")
    print("="*70)
    
    session_id = "resilience_session"
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if query.lower() == 'status':
                print("\n🔌 Circuit Breaker Status:")
                for tool_name, state in circuit_breaker.state.items():
                    failures = circuit_breaker.failures[tool_name]
                    print(f"   {tool_name}: {state} (failures: {failures})")
                continue
            
            if not query:
                continue
            
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "customer_id": "CUST001",
                "issue_type": "general",
                "errors": [],
                "retry_count": 0,
                "max_retries": 3,
                "iteration_count": 0,
                "max_iterations": 10,
                "degraded_mode": False,
                "fallback_reason": ""
            }
            
            config = {"configurable": {"thread_id": session_id}}
            result = agent.invoke(initial_state, config)
            
            # Show response
            final_msg = result["messages"][-1]
            print(f"\nAgent: {final_msg.content}")
            
            # Show stats
            if result.get('errors') or result.get('degraded_mode'):
                print(f"\n📊 Stats:")
                print(f"   Errors: {len(result.get('errors', []))}")
                print(f"   Retries: {result.get('retry_count', 0)}")
                print(f"   Degraded: {result.get('degraded_mode', False)}")
            
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
    print("🚀 DAY 2 AFTERNOON - ERROR HANDLING & RESILIENCE")
    print("="*70)
    print(f"📍 Using Azure deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    print("="*70)
    print("\nFeatures:")
    print("  ✅ Retry logic with exponential backoff")
    print("  ✅ Circuit breaker pattern")
    print("  ✅ Fallback to cached/degraded data")
    print("  ✅ Max iteration protection")
    print("  ✅ Graceful error handling")
    print("\nChoose mode:")
    print("  1. Run error handling tests")
    print("  2. Demo max iteration protection")
    print("  3. Interactive resilience testing")
    print("  4. Run all demos")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        test_error_handling()
    elif choice == "2":
        demo_max_iterations()
    elif choice == "3":
        interactive_resilience()
    else:
        print("\n🚀 Running all demos...\n")
        test_error_handling()
        input("\nPress Enter for next demo...")
        demo_max_iterations()
        input("\nPress Enter for interactive mode...")
        interactive_resilience()
