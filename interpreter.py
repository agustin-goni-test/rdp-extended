import os
from langgraph_setup import reactive_jql_app, JQLAnalysisState # Import your app and state
from jira_client import JiraClient

def main():

    test_tools()
#     # 1. Define the user query
#     initial_prompt = "find all the issues of type 'Historia' in project 'Equipo SVA' that have been solved this month"

#     # 2. Define the initial state
#     initial_state: JQLAnalysisState = {
#         "user_prompt": initial_prompt,
#         "suggested_jql": None,
#         "tool_calls": [],
#         "tool_results": [],
#         "validation_status": None,
#         "final_jql": None
#     }

#     print(f"--- Starting Agent for Prompt: {initial_prompt} ---")

#     # 3. Invoke the graph
#     # The graph will run the loop (Agent -> Tools -> Agent) until it hits END.
#     final_state = reactive_jql_app.invoke(initial_state)

#     # 4. Print the final result
#     print("\n--- Execution Complete ---")
#     print("Final Suggested JQL:")
#     # The agent's final output will be stored here
#     print(final_state.get('suggested_jql', 'JQL not generated.'))

#     print("\nHistory of Tool Calls and Results:")
#     for result in final_state.get('tool_results', []):
#         print(f"  - Tool Used: {result.tool_call_id} | Result: {result.content[:100]}...") # Truncate long results


def test_tools():

    # Crear cliente de Jira
    jira_client = JiraClient()

    # Prueba para encontrar nombre de proyecto    
    print("Probando herramienta de proyectos...")
    project = "Clientes"
    print(f"Buscando match con valor '{project}'...")
    match = jira_client.get_project_name_match(project)
    print(match)

    # Prueba de encontrar nombre de célula
    print("\n\nProbando herramienta de células...")
    team = "Adquirencia"
    print(f"Buscando match con valor '{team}...")
    match = jira_client.get_team_name_match(team)
    print(match)

    # Probar herramienta de tipos de issue
    
    print("Probando herramienta de tipo de issues...")
    issue = "Incidencia"
    print(f"Buscando match con valor '{issue}'...")
    match = jira_client.get_issue_type_name_match(issue)
    print(match)


if __name__ == "__main__":
    main()