import os
from langgraph_setup import reactive_jql_app, JQLAnalysisState # Import your app and state
from jira_client import JiraClient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from enricher import JQLEnrichmentAgent


load_dotenv()

api_key = os.getenv("LLM_API_KEY")
model = os.getenv("LLM_MODEL")

  
llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0
        )

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a Jira JQL generator. Convert the following natural language request into a valid Jira JQL query.

Request: {request}
JQL:
""")

parser = StrOutputParser()

# Compose the chain: prompt | llm | parser
jql_chain = prompt | llm | parser




def main():

    # test_tools()
    # 1. Define the user query
    # initial_prompt = "find all the issues of type 'Historia' in project 'Equipo SVA' that have been solved this month"

    # # 2. Define the initial state
    # initial_state: JQLAnalysisState = {
    #     "user_prompt": initial_prompt,
    #     "suggested_jql": None,
    #     "tool_calls": [],
    #     "tool_results": [],
    #     "validation_status": None,
    #     "final_jql": None
    # }

    # print(f"--- Starting Agent for Prompt: {initial_prompt} ---")

    # # 3. Invoke the graph
    # # The graph will run the loop (Agent -> Tools -> Agent) until it hits END.
    # final_state = reactive_jql_app.invoke(initial_state)

    # # 4. Print the final result
    # print("\n--- Execution Complete ---")
    # print("Final Suggested JQL:")
    # # The agent's final output will be stored here
    # print(final_state.get('suggested_jql', 'JQL not generated.'))

    # print("\nHistory of Tool Calls and Results:")
    # for result in final_state.get('tool_results', []):
    #     print(f"  - Tool Used: {result.tool_call_id} | Result: {result.content[:100]}...") # Truncate long results


    
    user_input = "give me all the issues of type 'Historia' or 'Componente Técnico' in project Equipo SVA that have been solved since the beginning of October 2025 and were assigned to either Edgar Benitez or Luis Vila. Order them by date of resolution"
    result = jql_chain.invoke({"request": user_input})
    print(f"Original JQL: ' {result} '")

    jira_client = JiraClient()

    agent = JQLEnrichmentAgent(llm, jira_client)

    enriched = agent.enrich(result)
    print(f"Enriched JQL: ' {enriched} '")

    # users = {"Agustín Goñi", "Edgar Benitez", "Luis Vila"}

    # jira_client = JiraClient()
    # list = jira_client.get_users_id(users)
    # print(list)


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