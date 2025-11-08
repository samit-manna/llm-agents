# 🎯 Day 2 Summary: Conditional Routing & Error Handling

## What You Built Today

### Morning: Smart Conditional Routing Agent
- ✅ LLM-based issue classification
- ✅ 4 specialized handlers (billing, tech, orders, general)
- ✅ Conditional edges based on classification
- ✅ Domain-specific tools per handler
- ✅ Confidence-based routing

### Afternoon: Production-Grade Error Handling
- ✅ Retry logic with exponential backoff
- ✅ Circuit breaker pattern
- ✅ Fallback to cached/degraded data
- ✅ Max iteration protection
- ✅ Graceful degradation

---

## 🎓 Key Concepts Mastered

### 1. Conditional Routing

**The Problem**
```python
# Bad: One agent tries to handle everything
agent.invoke("Check my payment")  # Uses wrong tools
agent.invoke("Reset password")     # Confused
agent.invoke("Track order")        # Inefficient
```

**The Solution**
```python
# Good: Route to specialized handlers
classify_issue() → "billing" → billing_handler (with billing tools)
classify_issue() → "technical" → tech_handler (with tech tools)
classify_issue() → "order" → order_handler (with order tools)
```

**Why It Matters**
- 🎯 **Accuracy**: Specialists handle their domain better
- ⚡ **Speed**: Fewer irrelevant tools = faster execution
- 💰 **Cost**: Smaller context windows = lower token costs
- 🔧 **Maintainability**: Easy to add new specialists

### 2. Circuit Breaker Pattern

**The Problem**
```
External API is down
Your agent keeps calling it
Every call times out (5 seconds)
100 users × 5 seconds = 500 seconds of wasted time
Users are frustrated
Your system is overloaded
```

**The Solution**
```python
States: CLOSED → OPEN → HALF_OPEN

CLOSED: Normal operation
  ↓ (3 failures)
OPEN: Stop calling (fast fail)
  ↓ (after 60s timeout)
HALF_OPEN: Try one call
  ↓ (success)
CLOSED: Back to normal
```

**Real-World Impact**
```
Without circuit breaker:
- 100 failed calls × 5s timeout = 500s wasted
- Users wait and get frustrated

With circuit breaker:
- 3 failed calls × 5s = 15s to detect
- Next 97 calls fail instantly (< 1ms)
- Total time: 15s vs 500s
- 97% time saved!
```

### 3. Retry with Exponential Backoff

**The Problem**
```
Tool fails → immediate retry → fails again
Hammering failing service makes it worse
Need intelligent retry strategy
```

**The Solution**
```python
Attempt 1: Call immediately
  ↓ (fails)
Attempt 2: Wait 1s, retry
  ↓ (fails)
Attempt 3: Wait 2s, retry
  ↓ (fails)
Attempt 4: Wait 4s, retry
  ↓ (success!)

Total time: 7s
But gave service time to recover
```

**When to Use**
- ✅ Transient network errors
- ✅ Rate limit errors
- ✅ Temporary service outages
- ❌ Validation errors (won't fix with retry)
- ❌ Authentication errors (need different fix)

### 4. Graceful Degradation

**The Problem**
```python
if payment_api_fails:
    crash()  # ❌ Bad: User gets nothing
```

**The Solution**
```python
if payment_api_fails:
    return cached_data  # ✅ Good: User gets something
    # "Unable to verify real-time, showing cached data from 2 hours ago"
```

**Degradation Hierarchy**
```
1. Real-time data (best)
   ↓ (if fails)
2. Recent cached data (good)
   ↓ (if unavailable)
3. Estimated data (acceptable)
   ↓ (if can't estimate)
4. Helpful error message (minimum)
```

### 5. Max Iteration Protection

**The Problem**
```python
# Agent enters infinite loop
while True:
    call_llm()  # Keeps thinking it needs more info
    call_tools()  # Tools return same data
    call_llm()  # Still not satisfied
    # ... forever ... 💸💸💸
```

**The Solution**
```python
MAX_ITERATIONS = 10

for i in range(MAX_ITERATIONS):
    result = agent_step()
    if done:
        break

if i >= MAX_ITERATIONS:
    return "Unable to complete after 10 attempts. Escalating to human."
```

**Why Critical**
- 💰 Prevents runaway costs
- ⚡ Ensures bounded latency
- 🛡️ Protects against infinite loops
- 👤 Escalates complex cases to humans

---

## 🏗️ Architecture Patterns

### Pattern 1: Classifier-Router-Handler

```
START
  ↓
CLASSIFIER (LLM decides issue type)
  ↓
ROUTER (Routes based on classification)
  ├→ BILLING_HANDLER (specialized tools)
  ├→ TECH_HANDLER (specialized tools)
  ├→ ORDER_HANDLER (specialized tools)
  └→ GENERAL_HANDLER (no special tools)
```

**Benefits:**
- Clear separation of concerns
- Easy to add new domains
- Testable components
- Auditable routing decisions

### Pattern 2: Try-Retry-Fallback

```
TRY: Call real-time API
  ↓ (fails)
RETRY: Wait + try again (3x)
  ↓ (still fails)
FALLBACK: Use cached data
  ↓ (no cache)
DEGRADE: Provide estimate
  ↓ (can't estimate)
ERROR: Helpful message
```

**When to Use Each:**
- Real-time: User needs latest data
- Retry: Transient failures expected
- Cached: Slightly stale is OK
- Estimate: Rough answer is useful
- Error: Be honest about limitations

### Pattern 3: Circuit Breaker + Fallback

```
Check circuit state
  ↓
CLOSED: Try normal call
  ↓ (success)
Return result ✅

CLOSED: Try normal call
  ↓ (3 failures)
OPEN circuit
  ↓
Use fallback immediately
  ↓
Wait timeout (60s)
  ↓
HALF_OPEN: Test with one call
  ↓ (success)
CLOSE circuit (recovered!)
```

---

## 📊 Production Metrics

### What to Track

**Routing Metrics**
```python
- Classification accuracy (% correct routes)
- Confidence scores (avg, p50, p95)
- Route distribution (billing: 40%, tech: 30%, etc)
- Misroutes requiring rerouting
```

**Error Metrics**
```python
- Error rate by type (timeout, connection, validation)
- Retry success rate (% resolved after retry)
- Circuit breaker trips (how often services fail)
- Degraded mode frequency (% requests in fallback)
- Mean time to recovery (after circuit opens)
```

**Performance Impact**
```python
Without retry: 95% success, 2s avg latency
With retry:    98% success, 2.5s avg latency

Trade-off:
+ 3% more successful requests
- 0.5s higher latency
= Worth it for critical operations!
```

---

## 💡 Production Best Practices

### 1. Classification Accuracy

```python
# Bad: Keyword matching
if "refund" in query:
    return "billing"

# Good: LLM classification with confidence
classification_result = llm.classify(query)
if classification_result.confidence < 0.6:
    return "general"  # Let general handler clarify
```

**Why:** 
- Keywords miss context ("Can I refund my friend?" != billing)
- Confidence scores prevent misroutes
- General handler can clarify ambiguous queries

### 2. Retry Budget

```python
# Set limits per request
MAX_RETRIES = 3
MAX_TOTAL_TIME = 10s

# Don't retry forever on the same error
if same_error_3_times:
    fail_fast()  # Something is fundamentally wrong
```

**Why:**
- Prevents wasting resources
- Gives predictable latency
- Detects systemic issues faster

### 3. Circuit Breaker Tuning

```python
# Don't be too sensitive
FAILURE_THRESHOLD = 5  # Not 1 or 2

# Don't wait forever
TIMEOUT = 60s  # Not 5 minutes

# Monitor recovery
track_circuit_state_changes()
alert_on_frequent_trips()
```

**Why:**
- Occasional failures are normal
- Quick recovery prevents long outages
- Frequent trips indicate bigger issues

### 4. Fallback Data Freshness

```python
# Tag cached data with timestamp
cached_data = {
    "data": payment_info,
    "cached_at": "2025-10-15T14:30:00Z",
    "age_hours": 2
}

# Be transparent
return f"Payment status (cached {age_hours}h ago): {data}"
```

**Why:**
- Users deserve to know data might be stale
- Helps them decide if they need to wait for real-time
- Builds trust

### 5. Error Messages for Users

```python
# Bad: Technical jargon
"ConnectionRefusedError: [Errno 111] Connection refused"

# Good: User-friendly + actionable
"We're having trouble connecting to our payment system. 
Your account is safe. Please try again in a few minutes 
or contact support if urgent."
```

**Why:**
- Users don't care about technical details
- They want to know: Is it my fault? What should I do?
- Good error messages reduce support tickets

---

## 🧪 Testing Strategies

### 1. Test Each Route

```python
@pytest.mark.parametrize("query,expected_route", [
    ("Check my invoice", "billing"),
    ("Reset my password", "technical"),
    ("Where's my order", "order_management"),
])
def test_routing(query, expected_route):
    result = agent.invoke(query)
    assert result.route == expected_route
```

### 2. Test Error Scenarios

```python
def test_retry_on_timeout():
    with mock.patch('api_call', side_effect=TimeoutError):
        result = resilient_tool.invoke()
        assert result.retry_count == 3
        assert result.used_fallback == True

def test_circuit_breaker():
    # Trigger failures
    for _ in range(5):
        with pytest.raises(Exception):
            tool.invoke()
    
    # Circuit should be open
    assert circuit_breaker.state == "OPEN"
```

### 3. Load Testing with Failures

```python
# Simulate 20% failure rate at scale
def load_test():
    results = []
    for i in range(1000):
        if random.random() < 0.2:
            inject_failure()
        
        result = agent.invoke(f"Query {i}")
        results.append(result)
    
    # Check resilience
    assert success_rate(results) > 0.95  # Still 95%+ despite failures
```

---

## 🎯 Day 2 vs Production

| Feature | Day 2 | Production |
|---------|-------|-----------|
| **Classification** | LLM-based | LLM + ML model fallback |
| **Retry attempts** | 3 | Configurable by service |
| **Circuit breaker** | Simple threshold | Per-service tuning |
| **Fallback data** | Mock | Redis cache + DB |
| **Max iterations** | Hard limit | Dynamic based on cost |
| **Error logging** | Print statements | Structured logs + Sentry |
| **Monitoring** | Manual observation | Grafana dashboards + alerts |

