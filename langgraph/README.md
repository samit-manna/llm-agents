# LangGraph Agent Framework Setup

> Experimentation workspace for building AI agents with LangGraph + Azure OpenAI

## 📁 Project Structure

```
langgraph/
├── .env                          # Environment variables (API keys)
├── requirements.txt              # Python dependencies
├── simple_agent.py              # Basic LangGraph agent with calculator tool
├── multi_tools_agent.py         # Agent with multiple tools (calculator + more)
├── agent_with_tavily_search.py  # Agent with web search capability
└── README.md                    # This file
```

## ✅ Current Setup Status

- ✓ Python virtual environment (`.venv` in parent directory)
- ✓ All packages installed
- ✓ Azure OpenAI configured
- ✓ LangSmith tracing enabled
- ✓ Tavily web search integrated

## 🔧 Configuration

All configuration is managed through `.env` file:

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-mini

# LangSmith (Tracing & Observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=llm-agents-langgraph

# Tavily (Web Search)
TAVILY_API_KEY=your-tavily-key
```

## 📦 Dependencies

```txt
langgraph              # Agent framework with graph-based workflows
langchain-openai       # Azure OpenAI & OpenAI integration
langchain-core         # Core LangChain functionality
langchain-community    # Community tools (Tavily search)
langsmith              # Observability and tracing
tavily-python          # Web search API
python-dotenv          # Environment variable management
jupyter                # Notebook support (optional)
```

## 🚀 Available Agents

### 1. Simple Agent (`simple_agent.py`)
Basic agent with a calculator tool - demonstrates fundamental LangGraph concepts.

**Features:**
- State management
- Tool binding and execution
- Conditional routing (use tool vs direct response)

**Run:**
```bash
python simple_agent.py
```

**Graph Flow:**
```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────┐
│  agent  │ ◄─────┐
└────┬────┘       │
     │            │
     ├─ No tool needed
     │            │
     ├─ Tool needed
     │            │
     ▼            │
┌─────────┐      │
│  tools  │──────┘
└─────────┘
     │
     ▼
┌─────────┐
│   END   │
└─────────┘
```

**Example Output:**
```
🧑 User: What is 123 * 456?
  🤔 Agent thinking...
  🔧 Agent decided to use tool: calculator
  🧮 Executing calculator with: 123 * 456
  ✓ Calculator result: 56088
  🤔 Agent thinking...
  💬 Agent answering directly (no tool needed)
🤖 Agent: 123 * 456 = 56,088
```

### 2. Multi-Tools Agent (`multi_tools_agent.py`)
Agent with multiple tools showcasing more complex decision-making.

**Run:**
```bash
python multi_tools_agent.py
```

**Graph Flow:**
```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────┐
│  agent  │ ◄─────┐
└────┬────┘       │
     │            │
     ├─ Decides which tool
     │            │
     ▼            │
┌─────────┐      │
│  tools  │──────┘
│ • Tool1 │
│ • Tool2 │
│ • Tool3 │
└─────────┘
     │
     ▼
┌─────────┐
│   END   │
└─────────┘
```

### 3. Agent with Web Search (`agent_with_tavily_search.py`)
Full-featured agent with calculator AND Tavily web search.

**Features:**
- Calculator tool for math operations
- Web search tool for current information
- Smart tool selection (agent decides which tool to use)
- Real-time execution logging

**Run:**
```bash
python agent_with_tavily_search.py
```

**Graph Flow:**
```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌──────────────┐
│    agent     │ ◄─────────┐
│  (decides)   │           │
└──────┬───────┘           │
       │                   │
       ├─ Calculator needed │
       │                   │
       ├─ Web search needed │
       │                   │
       ├─ Direct answer    │
       │                   │
       ▼                   │
┌──────────────┐           │
│    tools     │───────────┘
│              │
│ 🧮 calculator│
│ 🔍 web_search│
└──────────────┘
       │
       ▼
┌─────────┐
│   END   │
└─────────┘
```

**Example Outputs:**

**Math Query:**
```
🧑 User: What is 456 * 789?
  🤔 Agent thinking...
  🔧 Agent decided to use tool: calculator
  🧮 Executing calculator with: 456 * 789
  ✓ Calculator result: 359784
  🤔 Agent thinking...
  💬 Agent answering directly (no tool needed)
🤖 Agent: 456 * 789 = 359,784
```

**Web Search Query:**
```
🧑 User: What are the latest developments in AI agents in 2025?
  🤔 Agent thinking...
  🔧 Agent decided to use tool: web_search
  🤔 Agent thinking...
  💬 Agent answering directly (no tool needed)
🤖 Agent: [Comprehensive answer with latest AI agent developments...]
```

**General Knowledge (No Tool):**
```
🧑 User: What is the capital of Japan?
  🤔 Agent thinking...
  💬 Agent answering directly (no tool needed)
🤖 Agent: Tokyo.
```

**Example queries:**
- Math: "What is 456 * 789?"
- Search: "What are the latest developments in AI agents in 2025?"
- General: "What is the capital of Japan?"

## � Monitoring & Debugging

### Terminal Output
All agents print execution steps:
- 🤔 Agent thinking...
- 🔧 Agent decided to use tool: [tool_name]
- 🧮 Executing calculator with: [expression]
- 💬 Agent answering directly (no tool needed)

### LangSmith Dashboard
View detailed traces at: https://smith.langchain.com/

**What you'll see:**
- Visual execution graph
- Tool calls with inputs/outputs
- Token usage and timing
- Full message history
- Error traces

## � Key Concepts Learned

### 1. **State Management**
```python
class State(TypedDict):
    messages: Annotated[list, add]  # Messages accumulate
```

### 2. **Tool Definition**
```python
@tool
def calculator(operation: str) -> str:
    """Tool description for LLM"""
    return result
```

### 3. **Agent Node**
```python
def agent(state: State):
    llm = AzureChatOpenAI(...)
    llm_with_tools = llm.bind_tools([tool1, tool2])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### 4. **Conditional Routing**
```python
def should_continue(state: State):
    if last_message.tool_calls:
        return "tools"
    return END
```

### 5. **Graph Construction**
```python
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
```

## 🎯 Azure OpenAI vs Regular OpenAI

### Key Differences:

**Import:**
```python
# Azure
from langchain_openai import AzureChatOpenAI

# Regular OpenAI
from langchain_openai import ChatOpenAI
```

**Initialization:**
```python
# Azure
llm = AzureChatOpenAI(
    azure_deployment="your-deployment-name",
    api_version="2024-12-01-preview",
)

# Regular OpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
```

**Note:** Some Azure deployments don't support the `temperature` parameter.

## 🔧 Troubleshooting

### Temperature Error
```
Error: 'temperature' does not support 0.0 with this model
```
**Solution:** Remove the `temperature` parameter or use default value (1).

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables Not Loading
- Check `.env` file is in the correct directory
- Verify `load_dotenv()` is called at the top of your script

### Tool Not Being Used
- Check LangSmith traces to see decision-making
- Verify tool descriptions are clear and specific
- Ensure tool is bound to LLM: `llm.bind_tools([tool])`

## � Next Steps

1. ✅ **Basic Setup Complete** - You have working agents!
2. 🔄 **Add Memory** - Make agents remember conversation history
3. 🔀 **Multi-Agent Systems** - Create agents that delegate to sub-agents
4. 🛠️ **Custom Tools** - Build domain-specific tools
5. 🌐 **API Integration** - Connect to external APIs
6. 🎯 **Production Deployment** - Move from local to hosted

## 📖 Resources

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **LangSmith Dashboard:** https://smith.langchain.com/
- **Azure OpenAI Docs:** https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **Tavily Search API:** https://tavily.com/

---

**Status:** ✅ Setup verified and working  
**Last Updated:** October 16, 2025

