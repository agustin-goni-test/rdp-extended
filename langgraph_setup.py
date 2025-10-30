from typing import TypedDict, List, Union, Any
from langchain_core.tools import tool
from jira_client import JiraClient
from langchain_core.messages import ToolCall, ToolMessage as ToolResult
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()


# Assuming your JiraClient methods (get_all_projects, get_celula_dropdown_options, etc.)
# are now available as callable functions.

# The state must track the pending tool calls and results.
class JQLAnalysisState(TypedDict):
    user_prompt: str
    suggested_jql: Union[str, None] # Use Union[T, None] for Python < 3.10
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]
    validation_status: Union[str, None]
    final_jql: Union[str, None]


jira_client = JiraClient()

@tool
def get_all_projects():
    '''Retorna la lista de proyectos activos'''
    return jira_client.get_all_projects()

@tool
def get_all_issue_types():
    '''Retorna los tipos de issue que existen'''
    return jira_client.get_all_issue_types()

@tool
def get_celula_dropdown_options():
    '''Retorna las opciones de valor para célula'''
    return jira_client.get_celula_dropdown_options()

JIRA_TOOLS = [
        get_all_projects,
        get_all_issue_types,
        get_celula_dropdown_options
    ]

api_key = os.getenv("LLM_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                            api_key = api_key,
                            temperature=0
                        )
llm_with_tools = llm.bind_tools(JIRA_TOOLS)


# Define the Agent logic
def agent_node(state: JQLAnalysisState) -> JQLAnalysisState:
    print("\n" + "="*50)
    print("🤖 ENTERING AGENT NODE (Decision Maker)")
    print("="*50)

    # 1. Determine the status of the conversation history
    # This flag is the CRITICAL logical switch for the LLM's behavior
    history_exists = "TRUE" if state.get('tool_results') else "FALSE"
    
    # 2. Format the tool history for the LLM
    tool_history = [
        ("user", f"Tool Results: {r.content}")
        for r in state.get('tool_results', [])
    ]

    # --- System Prompt Definition ---
    system_prompt = (
        "You are a highly capable JQL query generator. Your sole mission is to produce a single, valid JQL string that directly answers the user's request. The JQL field for 'Celula' is always written as '\"Celula[Dropdown]\"'.\n"
        "\n"
        "**CRITICAL LOGIC:**\n"
        "**- IF history_exists is FALSE, you MUST respond ONLY with tool calls.**\n"
        "**- IF history_exists is TRUE, you MUST respond ONLY with the final JQL.**\n"
        "\n"
        "**PHASE 1: DATA GATHERING & VALIDATION (Use Tools)**\n"
        "1. MANDATORY VALIDATION: Before generating any JQL, you **MUST** use the provided tools to validate every entity (Project names, Issue Types, Celula values) mentioned in the user's request. Perform all required checks in the minimum number of tool calls possible.\n"
        "\n"
        "**PHASE 2: FINAL JQL GENERATION (Stop Condition)**\n"
        "2. CRITICAL STOP CONDITION: If history_exists is TRUE, immediately cease all tool calls. Construct the final JQL based on the validated data and the user's request.\n"
        "3. OUTPUT FORMAT: Your final response must contain ONLY the valid JQL string and nothing else (no markdown, commentary, or leading phrases). The JQL must be the very first and last thing you output.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        
        # --- NEW CONTEXT FLAG ---
        ("system", f"Context Flag: history_exists={history_exists}"),

        # --- JQL SYNTAX HINTS (Escaped Braces) ---
        ("system", 
         "**JQL SYNTAX HINTS:**\n"
         "* Dates are relative: Use the JQL `resolutiondate >= startOfMonth{{}}()`, `resolutiondate >= '-3M'`, or similar syntax.\n"
         "* 'Has solved' or 'resolved' implies using the `resolutiondate` field.\n"
         "* If the project is not specified, use `project IN ('SW', 'LA')` as a default filter."
        ),
        
        # --- FEW-SHOT EXAMPLES (Escaped Braces) ---
        ("user", "Find issues solved by Alpha Team."),
        ("assistant", "get_celula_dropdown_options{{}}()"),
        ("user", "Tool Results: ['Alpha Team', 'Beta Team']"),
        ("assistant", "project IN (SW, LA) AND resolution IS NOT EMPTY AND \"Celula[Dropdown]\" = 'Alpha Team'"),

        ("user", "All bugs resolved this month in the new project."),
        ("assistant", "get_all_projects{{}}() AND get_all_issue_types{{}}()"),
        ("user", "Tool Results: [Projects: {{SW: 'Service Web'}}, Types: {{Bug, Task}}]"),
        ("assistant", "project = SW AND issuetype = Bug AND resolutiondate >= startOfMonth{{}}()"),

        # --- USER'S CURRENT QUERY ---
        ("user", "{user_prompt}"),
        ("assistant", "{tool_history}"),
    ])

    # 3. Create the runnable chain and invoke
    agent_runnable = prompt | llm_with_tools
    
    # Note: Removed the retry logic for clarity, but keep it if rate limits are an issue.
    response = agent_runnable.invoke({
        "user_prompt": state["user_prompt"],
        "tool_history": tool_history,
        "history_exists": history_exists # Pass the flag to the runnable
    })

    # 4. Process the LLM's output (Decision Point)
    if response.tool_calls:
        print(f"➡️ AGENT DECISION: Calling {len(response.tool_calls)} Tool(s). Moving to 'tools' node.")
        print("="*50)
        return {"tool_calls": response.tool_calls}
    else:
        print("✅ AGENT DECISION: Final JQL Generated. Moving to 'END' node.")
        print("="*50)
        return {"suggested_jql": response.content}


def tool_execution_node(state: JQLAnalysisState) -> JQLAnalysisState:

    print("\n" + "#"*50)
    print(f"🛠️ ENTERING TOOLS NODE (Executing {len(state['tool_calls'])} calls)")
    print("#"*50)
    
    tool_calls = state["tool_calls"]
    tool_results = []
    
    # Map the tool name (string) back to the actual Python function
    tools_map = {tool.name: tool for tool in JIRA_TOOLS}

    for call in tool_calls:
        tool_name = call.get("name")
        tool_args = call.get('args', {})
        call_id = call.get('id')

        result = ""
        
        print(f"   * EXECUTING: {tool_name}({tool_args})")
        
        if tool_name not in tools_map:
            result = f"Error: Tool {tool_name} not found."
        else:
            try:
                # Execute the corresponding function (e.g., get_all_projects())
                tool_function = tools_map[tool_name]
                
                # We assume the tools take no arguments based on your current setup.
                output = tool_function(**tool_args)
                result = str(output) # Convert list/dict output to a string for the LLM
                
            except Exception as e:
                result = f"Tool execution failed: {e}"

            print(f"   * RESULT COLLECTED (Length: {len(result)}).")

        # Store the result for the agent to use in the next loop
        tool_results.append(ToolResult(
            tool_call_id=call_id,
            content=result
        ))

    print("#"*50)
    print("↩️ TOOLS EXECUTION COMPLETE. Moving back to 'agent' node.")
    print("#"*50)
        
    return {
        "tool_results": tool_results,
        "tool_calls": [], # Clear tool_calls to signal the action is complete
        "validation_status": "TOOLS_EXECUTED"
    }


# Assume agent_node and tool_execution_node are defined using the agent pattern

# 1. Initialize the builder
workflow = StateGraph(JQLAnalysisState)

# 2. Add the nodes
workflow.add_node("agent", agent_node) # The LLM decision-maker
workflow.add_node("tools", tool_execution_node) # Executes JiraClient methods

# 3. Set the entry point
workflow.set_entry_point("agent")

# 4. Define the Tool-Use Cycle (The Loop)
workflow.add_conditional_edges(
    "agent",
    # The output from the agent_node determines the next step:
    # This function checks if the agent requested a tool call or provided the final JQL.
    lambda state: "tools" if state.get("tool_calls") else "end",
    {
        "tools": "tools", # If tool_calls exist, go to the tool_execution_node
        "end": END        # Otherwise, the agent has provided the final JQL (or failed), so end.
    }
)

# 5. Tool Execution must always return control to the agent to decide the next step
workflow.add_edge("tools", "agent")

# 6. Compile the graph
reactive_jql_app = workflow.compile()