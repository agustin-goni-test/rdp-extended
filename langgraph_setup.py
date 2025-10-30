from typing import TypedDict, List, Union, Any
from langchain_core.tools import tool
from jira_client import JiraClient
from langchain_core.messages import ToolCall, ToolMessage as ToolResult
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, ToolMessage
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
def get_project_name_match():
    '''Encuentra la mejor aproximación para el nombre de un proyecto'''
    return jira_client.get_project_name_match()

@tool
def get_issue_type_name_match():
    '''Encuentra la mejor aproximación para un tipo de issue'''
    return jira_client.get_issue_type_name_match()

@tool
def get_celula_dropdown_name():
    '''Encuentra la mejor aproximación para el nombre de una célula'''
    return jira_client._get_celula_dropdown_options()

JIRA_TOOLS = [
        get_project_name_match,
        get_issue_type_name_match,
        get_celula_dropdown_name
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
    tool_calls = state.get('tool_calls', []) # assuming you store these after the first turn
    tool_results = state.get('tool_results', [])

    history_messages = []

    # We must iterate over the successful tool calls and provide the result
    for call in tool_calls:
        # 1. Assistant's action message (The tool call)
        history_messages.append(AIMessage(content="", tool_calls=[call])) 
        
        # 2. Tool's observation message (The result of the action)
        # Find the result that matches this tool call ID
        result = next((r for r in tool_results if r.tool_call_id == call.id), None)
        
        if result:
            history_messages.append(ToolMessage(
                content=result.content, 
                tool_call_id=call.id,
                # name=call.name # Optional, but recommended for clarity
            ))

    tool_history = format_tool_history(tool_results, tool_calls)

    # --- System Prompt Definition ---
    # system_prompt = '''
    #     Tu objetivo es entregar una consulta en JQL que pueda implementar un filtro y generar una lista de issues de Jira que cumplan con
    #     alguna condición.

    #     Para lograr esto, primero debes entender que es lo que busca el usuario, y validar ciertas entidades de Jira para obtener la consulta. Para
    #     esto puedes usar las herramientas que tienes disponibles.

    #     En particular, considera que la solicitud del usuario puede mencionar lo siguiente:
    #     - 'Proyecto' o similar, que se refiere al campo 'Project' de Jira, y que puedes validar con la herramienta get_project_name_match().
    #     - 'Equipo', 'célula' o similar, que se refiere al campo 'Celula[Dropdown]', y hace referencia al equipo encargado. Lo puedes validar con get_celula_dropdown_name().
    #     - 'Tipo', 'tipo de issue', 'tipo de ticket', se refiere al campo 'type' de Jira. Para validar usa get_issue_type_name_match(). Ten en cuenta que en
    #     la conversación esta entidad puede ir implícita (por ejemplo, el usuario puede decir 'todos los incidentes resueltos', en vez de 'todos los issues de tipo incidente').
    #     - Preguntas por fechas. Esas son fundamentales, y muchas veces relativas (últimos 3 meses, desde el inicio de mes, etc). Para esto puedes usar expresiones tales como startOfMonth.

    #     Tu mecanismo de respuesta es el siguiente:
    #     1. Determina las posibles entidades a las que se refiere el usuario (proyectos, equipos, tipos de issue, etc).
    #     2. Usa las herramientas para identificar las instancias exactas, o al menos aproximadas (por ejemplo, el proyecto con un nombre similar).
    #     3. Define si hay restricciones por fecha.
    #     4. Generar un JQL que capture esta necesidad.

    #     La idea es que puedas hacerlo con la menor cantidad de llamadas posibles, pero manteniendo la coherencia.

    #     Ejemplo 1: 'buscar todos los issues resueltos por el Equipo de Web Privada desde el 1 de julio de este año'.
    #     "acción": usar herramienta "get_celula_dropdown_name{{}}()"
    #     "resultado":  "['name'= 'Web Privada', 'confidence': 100.0]"
    #     "acción": Contamos con el nombre del equipo. Dado que también conocemos las fechas y no hay más entidades involucradas podemos terminar.
    #     "resultado": JQL 'CelulaCelula[Dropdown] = 'Equipo SVA' and resolved > '2025-07-01'

    #     "Ejemplo 2: "buscar todos las incidencias resueltas en el proyecto SVA durante este mes'    
    #     "acción": usar herramienta get_project_name_match {{}}() y get_issue_type_name_match {{}}()
    #     "resultado": Proyecto ['key' = 'WPRI', 'name'= 'Web Privada', 'confidence': 100.0] y Tipo ['id' = 1929, 'name'= 'Incidente', 'confidence': 100.0]
    #     "acción": identificamos el proyecto y el tipo de issue. Además sabemos que debemos usar una consulta relativa desde principios del mes.
    #     "resultado": JQL 'Project = 'Equipo SVA' AND type = 'Incidente' and resolved > startOfMonth() ORDER BY resolved'
    # '''

    system_prompt = '''
    You are a highly capable **JQL query generator** for Jira. Your **sole output** must be a valid JQL string or a function call.

    ---
    ### 🎯 CORE DIRECTIVE
    Your output must be **one of two things**:
    1.  **A Tool Call:** If project, issue type, or team names need validation.
    2.  **The Final JQL String:** If all necessary data is validated OR no more filters are needed.

    Your final JQL output **MUST NOT** contain markdown, commentary, or extra text.

    ### ⚙️ FIELD MAPPING & RULES
    * **Team/Celula:** Use `"Celula[Dropdown]"`. Validate with `get_celula_dropdown_name()`.
    * **Project:** Use `project`. Validate with `get_project_name_match()`.
    * **Issue Type:** Use `type`. Validate with `get_issue_type_name_match()`.
    * **Resolved/Solved:** Always add `resolution IS NOT EMPTY`.
    * **Default Project:** If the user doesn't specify a project, default to `project IN ('SW', 'LA')`.

    ---
    ### ✅ WORKFLOW EXAMPLES (Teaching the Loop Logic)

    **EXAMPLE 1: Requires Tool Call (PHASE 1)**
    **USER:** Find all resolved incidents from Team Beta.
    **ASSISTANT:** get_issue_type_name_match() AND get_celula_dropdown_name()

    **TOOL RESULTS:** [Type: {'name': 'Incident', 'confidence': 100.0}, Team: {'name': 'Team Beta', 'confidence': 95.0}]

    **ASSISTANT (PHASE 2: Final JQL Generation):**
    Project IN ('SW', 'LA') AND type = Incident AND Celula[Dropdown] = 'Team Beta' AND resolution IS NOT EMPTY


    **EXAMPLE 2: Requires Date Logic (No Tools Needed)**
    **USER:** All issues from project 'SW' created in the last 3 months.
    **ASSISTANT (PHASE 2: Final JQL Generation):**
    project = SW AND created >= "-3M"

    '''

    prompt = ChatPromptTemplate.from_messages([
        # 1. The Core Instructions, Rules, and Few-Shot Examples
        ("system", system_prompt),
        
        # 2. The history of the current loop (Tool Call and Tool Result)
        # This is how the LLM sees the output of its last action.
        MessagesPlaceholder(variable_name="tool_history"), 
        
        # 3. The original request, always presented as the last human message.
        ("user", "{user_prompt}"), 
    ])

    # 3. Create the runnable chain and invoke
    agent_runnable = prompt | llm_with_tools
    
    # Note: Removed the retry logic for clarity, but keep it if rate limits are an issue.
    response = agent_runnable.invoke({
        "user_prompt": state["user_prompt"],
        "history_messages": history_messages,
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


def format_tool_history(tool_results: list, tool_calls: list) -> list:
    history = []
    # Loop through the executed tool calls and results, pairing them up
    for tool_call in tool_calls:
        # 1. Assistant's message that made the tool call
        history.append(AIMessage(content="", tool_calls=[tool_call]))
        
        # Find the matching tool result
        result = next((r for r in tool_results if r.tool_call_id == tool_call.id), None)
        if result:
            # 2. The tool's observation (The data the agent uses to decide the JQL)
            history.append(ToolMessage(content=result.content, tool_call_id=tool_call.id))
    return history