from typing import Any, Callable
from langchain_core.tools.base import BaseTool
from google.genai.types import Tool
import streamlit as st
import os
import concurrent.futures
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import urllib.request
import urllib.parse

load_dotenv()
def clean_llm_output(content):
    """Extracts just the text from Gemini's complex response objects."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # If it's a list of blocks, extract the 'text' fields and join them
        return " ".join([block.get("text", "") for block in content if isinstance(block, dict) and "text" in block])
    return str(content)
# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise AI Orchestrator", page_icon="⚙️", layout="wide")
st.title("⚙️ Multi-Tool AI Orchestrator (Production Mode)")

# --- FILE I/O HELPERS (MOCK DATABASE) ---
# This ensures we don't crash if the files don't exist yet
def append_to_db(filename, content):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content + "\n")

def read_db(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return "Database empty."


# --- LOAD BACKEND TOOLS (WITH ERROR HANDLING) ---
@tool
def schedule_meeting(title: str, day: str, time: str, duration: str) -> str:
    """Schedules a meeting. If the user does not provide the specific time or duration, DO NOT GUESS. Ask them for it."""
    # PATH 3: Error Handling & Refusal
    if time.lower() in ["unknown", "none", "tbd", ""] or duration.lower() in ["unknown", "none", ""]:
        return "ERROR: Missing time or duration. Tell the user you need the exact time and duration before scheduling."
    
    # PATH 2: Real File I/O
    record = f"📅 {day} @ {time} | {title} ({duration})"
    append_to_db("calendar_db.txt", record)
    
    return f"Success: Wrote '{record}' to calendar database."

@tool
def save_reminder(task_description: str, priority: str = "medium") -> str:
    """Saves a reminder or note. Priority defaults to medium unless specified."""
    # PATH 2: Real File I/O
    record = f"📌 [{priority.upper()}] {task_description}"
    append_to_db("reminders_db.txt", record)
    
    return f"Success: Saved '{task_description}' to reminders database."

# --- TOOL 3: REST API INTEGRATION ---
@tool
def fetch_live_weather(city: str) -> str:
    """Fetches the current live weather for a specified city. Use this whenever the user asks about the weather."""
    try:
        # Encode the city name for a URL and fetch from the free wttr.in API
        safe_city = urllib.parse.quote(city)
        url = f"https://wttr.in/{safe_city}?format=%C+%t"
        
        # We use a custom User-Agent because some APIs block standard Python scripts
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            weather_data = response.read().decode('utf-8')
            
        return f"Success: The current weather in {city} is {weather_data.strip()}."
    except Exception as e:
        return f"ERROR: Could not fetch weather API for {city}. Details: {e}"

# --- TOOL 4: LOGIC & MATH COMPUTE ---
@tool
def evaluate_logic_expression(expression: str) -> str:
    """Evaluates mathematical equations or boolean logic (e.g., '10 * 5', 'True and False'). Use this instead of guessing math!"""
    try:
        # A highly restricted eval() environment to safely calculate numbers or logic
        allowed_names = {"True": True, "False": False, "abs": abs, "min": min, "max": max}
        
        # Process the logic string
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Success: The logic expression '{expression}' evaluates to: {result}"
    except Exception as e:
        return f"ERROR: Invalid logic expression '{expression}'. Tell the user to check their syntax."
tools_registry = {
    "schedule_meeting": schedule_meeting,
    "save_reminder": save_reminder,
    "fetch_live_weather": fetch_live_weather,
    "evaluate_logic_expression": evaluate_logic_expression
}

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(list[dict[str, Any] | type | Callable[..., Any] | BaseTool | Tool](tools_registry.values()))

# --- INITIALIZE SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# --- PATH 1: LIVE TELEMETRY & STATE INSPECTOR (SIDEBAR) ---
with st.sidebar:
    st.header("🔍 System Telemetry")
    
    st.subheader("🗄️ Active Databases")
    st.caption("calendar_db.txt")
    st.code(read_db("calendar_db.txt"), language="text")
    
    st.caption("reminders_db.txt")
    st.code(read_db("reminders_db.txt"), language="text")
    
    st.divider()
    
    st.subheader("🧠 Memory Stack")
    st.caption(f"Current Context Window: {len(st.session_state.chat_history)} messages")
    with st.expander("View Raw Payload Data", expanded=False):
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.success(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    st.warning(f"AI Tool Request: {msg.tool_calls}")
                else:
                    st.info(f"AI: {msg.content}")
            elif isinstance(msg, ToolMessage):
                st.error(f"Tool Result: {msg.content}")

# --- RENDER PAST CHAT HISTORY ---
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def execute_tool_logic(tool_call):
    """Runs the tool in a background thread and returns the result payload."""
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    call_id = tool_call['id']
    
    if tool_name in tools_registry:
        result = tools_registry[tool_name].invoke(tool_args)
        status_flag = "error" if "ERROR" in result else "success"
        return {"name": tool_name, "args": tool_args, "result": result, "id": call_id, "status": status_flag}
    else:
        return {"name": tool_name, "args": tool_args, "result": f"Error: Tool '{tool_name}' not found.", "id": call_id, "status": "error"}

# --- CHAT INPUT INTERACTION ---
if user_input := st.chat_input("Enter your message here"):
    
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.ui_messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    with st.chat_message("assistant"):
        with st.spinner("Processing intent..."):
            response = llm_with_tools.invoke(st.session_state.chat_history)
            st.session_state.chat_history.append(response)
            
            if response.tool_calls:
                # Create a single UI status box for the parallel batch
                with st.status(f"Executing {len(response.tool_calls)} tools in parallel...", expanded=True) as status:
                    
                    # Open a Thread Pool to run tasks concurrently
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Dispatch all tool calls to the background threads instantly
                        future_to_tool = {executor.submit(execute_tool_logic, tc): tc for tc in response.tool_calls}
                        
                        # As each thread finishes, catch the result and update the main UI
                        for future in concurrent.futures.as_completed(future_to_tool):
                            data = future.result()
                            
                            st.write(f"**Payload for {data['name']}:** {data['args']}")
                            
                            if data["status"] == "error":
                                st.error(f"❌ **Validation Failed:** {data['result']}")
                            else:
                                st.success(f"✅ **Result:** {data['result']}")
                                
                            # Append the tool's result to the memory stack
                            st.session_state.chat_history.append(ToolMessage(content=data['result'], tool_call_id=data['id']))
                            
                    status.update(label="Parallel execution complete!", state="complete")
                            
                # 4. Resolve the final state now that all threads are done
                with st.spinner("Resolving state..."):
                    final_response = llm_with_tools.invoke(st.session_state.chat_history)
                    st.session_state.chat_history.append(final_response)
                    
                    clean_text = clean_llm_output(final_response.content)
                    st.markdown(clean_text)
                    st.session_state.ui_messages.append({"role": "assistant", "content": clean_text})
                    st.rerun()