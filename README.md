# ⚙️ Enterprise AI Multi-Tool Orchestrator

A stateful, multi-tool orchestration engine built with Python, LangChain, and Streamlit. This application demonstrates advanced LLM control flows, including concurrent tool execution, multi-turn memory management, and custom execution guardrails.

## 🚀 Key Features

* **Concurrent Tool Execution:** Utilizes Python's `ThreadPoolExecutor` to dispatch and resolve multiple LLM-requested tool calls simultaneously, drastically reducing pipeline latency.
* **Direct Tool Binding (ReAct Architecture):** Bypasses monolithic agent abstractions by explicitly binding Python functions to the LLM, allowing for dynamic routing based on natural language intent.
* **Multi-Turn Context State:** Implements a rigorous in-memory stack using `HumanMessage`, `AIMessage`, and `ToolMessage` payloads to maintain conversation context across complex workflows.
* **Execution Guardrails & Validation:** Intercepts LLM tool calls to validate arguments before execution, forcing the AI to ask the user for clarification rather than hallucinating missing inputs.
* **Live Telemetry & State Inspector:** Features a real-time Streamlit dashboard to monitor the active context window, view raw JSON payload routing, and observe external database states.
* **Local Data Persistence:** Simulates external system state by dynamically reading and writing to local databases with UTF-8 encoding compliance.
* **Third-Party API Integration:** Safely constructs external REST calls (e.g., live weather fetching) and parses live data back into the LLM's context stream.

## 🏗️ Technical Architecture

* **Frontend:** Streamlit web interface with dynamic state containers.
* **Backend Engine:** Python, LangChain Core, `concurrent.futures`.
* **LLM:** Google Gemini 2.5 Flash via `langchain-google-genai`.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/pranaypalaboyina23-cell/ai-orchestrator.git](https://github.com/pranaypalaboyina23-cell/ai-orchestrator.git)
   cd ai-orchestrator

2. Set up the virtual environment:
   ```bash
   python -m venv venv
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate

3. Install dependencies:
   ```bash
    pip install streamlit langchain langchain-core langchain-google-genai python-dotenv

4. Environment Variables:
    Create a .env file in the root directory and add your Google Gemini API key:

Code snippet
GOOGLE_API_KEY=your_api_key_here

💻 Usage
Run the Streamlit application locally:

   ```bash
    streamlit run app.py