import os
from langgraph_setup import reactive_jql_app, JQLAnalysisState # Import your app and state
from jira_client import JiraClient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from enricher import JQLEnrichmentAgent


load_dotenv()

# Config params
api_key = os.getenv("LLM_API_KEY")
model = os.getenv("LLM_MODEL")

# LLM basado en modelos de Gemini
llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0
        )

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a Jira JQL generator. Convert the following natural language request into a valid Jira JQL query. Take certain things into consideration:
 - the active sprint of a team can be obtained with the function openSprints() in JQL.
- be mindful of present and past tenses. For example, if required to provide issues concerning an assignee, separate the case in which the 
issue is assigned to that person ("assignee" in or "assignee =") to the case were it was once assigned ("assigne WAS")
- if the prompt mentions a team, as in "team Clientes" or "cell APM", it refers to a Jira field called "Celula[Dropdown]", so an expression like 'team Adquirencia' becomes ' Celula[Dropdown] = "Adquirencia" ' 

Request: {request}
JQL:
""")

# Crear parser para la salida
parser = StrOutputParser()

# Compose the chain: prompt | llm | parser
jql_chain = prompt | llm | parser




def main():
    
    # user_input = "give me all the issues of type 'Historia' or 'Componente Técnico' in project Equipo SVA that have been solved since the beginning of October 2025 and were assigned to either Edgar Benitez or Luis Vila. Order them by date of resolution"

    # user_input = "give me all issues of type Historia in project Equipo SVA that belong to the current active sprint and were once assigned to Edgar Benitez"
    user_input = "give me all issues of type Incidente in from teams Clientes, Afiliación y Contratos, Web Privada or Web Publica that have been solved or created this year"

    # Invocar cadena para obtener la expresión JQL cruda
    result = jql_chain.invoke({"request": user_input})
    print(f"Original JQL: ' {result} '")

    jira_client = JiraClient()

    # Crear agente para enriquecimiento de JQL e invocar. Usa herramientas
    agent = JQLEnrichmentAgent(llm, jira_client)
    enriched = agent.enrich(result)
    print(f"Enriched JQL: ' {enriched} '")

    # Mostrar los issues de resultado
    issues = jira_client.get_issues_from_jql(enriched)
    for issue in issues:
        print(issue.key)


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