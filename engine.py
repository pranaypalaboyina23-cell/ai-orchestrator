import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

# --- TOOLS REGISTRY ---
@tool
def schedule_meeting(title: str, day: str, time: str, duration: str) -> str:
    """Schedules a meeting on the calendar. Use this whenever the user explicitly asks to book or schedule a meeting."""
    print(f"\n[TOOL EXECUTION: Writing to Calendar Database...]")
    return f"Success: Scheduled '{title}' on {day} at {time} for {duration}."

@tool
def save_reminder(task_description: str, priority: str = "medium") -> str:
    """Saves a standalone reminder, task, or note. Use this when the user wants to remember something, log a note, or add a to-do item."""
    print(f"\n[TOOL EXECUTION: Writing to Reminders Database...]")
    return f"Success: Saved reminder '{task_description}' with {priority} priority."

tools_registry = {
    "schedule_meeting": schedule_meeting,
    "save_reminder": save_reminder
}

# --- INITIALIZE LLM WITH BINDINGS ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(list(tools_registry.values()))

# --- IN-MEMORY SESSION HISTORY ---
# This list will hold the ongoing chain of messages (Human, AI, and Tool feedback)
chat_history = []

def run_orchestrator(user_prompt: str):
    """Processes user input, updates session memory, handles dynamic tool execution loops."""
    global chat_history
    
    # Append the new user input to the rolling context window
    chat_history.append(HumanMessage(content=user_prompt))
    
    # Send the entire conversation history to the model
    response = llm_with_tools.invoke(chat_history)
    
    # Append the model's response (which could be text or tool call obligations)
    chat_history.append(response)
    
    # If the LLM requested tool executions
    if response.tool_calls:
        print(f"\n[Orchestrator: LLM identified {len(response.tool_calls)} action(s)]")
        
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            call_id = tool_call['id']
            
            print(f" -> Routing to Tool: '{tool_name}' with arguments: {tool_args}")
            
            if tool_name in tools_registry:
                # Run the function
                tool_result = tools_registry[tool_name].invoke(tool_args)
                print(f" -> {tool_result}")
                
                # CRITICAL STEP: Feed the tool's execution confirmation back into memory
                # This lets the model know the action was successfully taken.
                chat_history.append(ToolMessage(content=tool_result, tool_call_id=call_id))
            else:
                error_msg = f"Error: Tool '{tool_name}' not found."
                print(f" -> {error_msg}")
                chat_history.append(ToolMessage(content=error_msg, tool_call_id=call_id))
        
        # Second invocation: Let the LLM examine the tool output and generate a final human-readable conclusion
        final_response = llm_with_tools.invoke(chat_history)
        chat_history.append(final_response)
        print(f"\nAI Response: {final_response.content}")
        
    else:
        # Standard conversation if no tools were necessary
        print(f"\nAI Response: {response.content}")


# --- INTERACTIVE USER LOOP ---
if __name__ == "__main__":
    print("=" * 60)
    print("AI MULTI-TOOL ORCHESTRATION ENGINE WITH MEMORY STATE")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down engine. Goodbye!")
                break
                
            run_orchestrator(user_input)
            
        except Exception as e:
            print(f"\nAn error occurred during execution: {e}")