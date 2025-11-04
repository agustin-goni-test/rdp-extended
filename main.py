import os
from dotenv import load_dotenv
from jira import JIRA
from datetime import datetime
from jira_client import JiraClient, IssueInfo, IssueAnalysis
from typing import List
from llm_client import LLMClient
from business_info import BusinessInfo
from output_manager import OutputManager, OutputRunnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
import asyncio
import logging
from limiter import RateLimitingRunnable
from enricher import JQLEnrichmentAgent
from logger import Logger

load_dotenv()

JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN")
FILTER_ID = os.getenv("JIRA_FILTER_ID")
EXECUTION = os.getenv("EXECUTION")
RATE_LIMITING = True if os.getenv("RATE_LIMITING") == "true" else False


def main():
    print("Everything OK!")

    logger = Logger()
    logger.info("Comenzando proceso...")
    
    # Expresión en lenguaje natural de entrada para obtener los issues
    user_input = "give me all issues of type Historia or Componente Técnico in project Equipo SVA that were solved between October 1 and October 31 of 2025, in status Finalizado and were once assigned to Edgar Benitez, Luis Vila or Alexis Apablaza"

    start_time = datetime.now()

    # Leer filtro según el código pre definido
    filter = os.getenv("JIRA_FILTER_ID")

    # Buscar información de los issues del filtro
    # issues_info = get_issue_list_info(filter)
    logger.info("Obteniendo issues a través del LLM...")
    issues_info = get_issue_list_info_llm(user_input)

    # Generar la salida, pudiendo ser de forma síncrona o asíncrona
    logger.info("Crear tabla de salida...")
    if EXECUTION == "asynch":
        logger.info("Ejecutaremos en forma ASÍNCRONA...")
        asyncio.run(create_output_table_async(issues_info))
    else:
        create_output_table(issues_info)

    # Medir tiempo de ejecución total
    finish_time = datetime.now()
    total_seconds = (finish_time - start_time).total_seconds()
    elapsed_minutes = int(total_seconds // 60)
    elapsed_seconds = int(total_seconds % 60)
 
    logger.info(f"Proceso terminado en {elapsed_minutes} minutos y {elapsed_seconds} segundos")


def get_issue_list_info(filter) -> List[IssueInfo]:
    '''
    Método para obtener la información de los issues desde un filtro de Jira
    '''
    # Instanciar cliente Jira
    jira_client = JiraClient()
    
    #Obtener los issues desde el filtro
    issues = jira_client.get_issues_from_filter(filter)
    
    # Capturar información de cada uno de los issues
    info = jira_client.proccess_issue_list_info(issues)

    return info



def get_issue_list_info_llm(user_input) -> List[IssueInfo]:
    '''Método para obtener información de issues a partir de una expresión en lenguaje
    natural.
    
    Implementa un grafo de enriquecimiento para manipular la salida JQL'''
    
    # Crear cliente de Jira y logger
    jira_client = JiraClient()
    logger = Logger()

    # Obtener parámetros de configuración
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("LLM_API_KEY")

    # Crear handler de LLM
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0
        )
    
    # Crear agente de enriquecimiento de JQL
    agent = JQLEnrichmentAgent(llm, jira_client)
    
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

    # user_input = "give me all issues of type Historia in project Equipo SVA that belong to the current active sprint and were once assigned to Edgar Benitez"

    # Instanciar parser de salida para la expresión
    parser = StrOutputParser()

    # Componer la cadena: prompt | llm | parser
    jql_chain = prompt | llm | parser

    # Generar expresión cruda de SQL
    expression = jql_chain.invoke({"request": user_input})

    # Enriquecer expresión de SQL
    enriched_expression = agent.enrich(expression)
    logger.info(f"Expresión JQL a usar: {enriched_expression}")

    # Obtener información de los issues relacionados
    info = get_issue_list_info_from_jql(enriched_expression)

    return info

def get_issue_list_info_from_jql(jql) -> List[IssueInfo]:
    '''
    Método para obtener la información de los issues desde un filtro de Jira
    '''

    # Instanciar cliente Jira
    jira_client = JiraClient()
    
    #Obtener los issues desde el filtro
    issues = jira_client.get_issues_from_jql(jql)
    
    # Capturar información de cada uno de los issues
    info = jira_client.proccess_issue_list_info(issues)

    return info


def get_business_info() -> str:
    '''
    Método para obtener la información de negocio relativa a una HU.
    '''
    # Usar cliente de información del negocio (genérico)
    business_info = BusinessInfo()

    # Obtener la información a través de un método del cliente
    info = business_info.get_business_info("GOBI-895")

    # Retornar la información
    return info


# Método "histórico". Ya no estamos usando llamadas directas a un cliente LLM
def test_llm_client():
    '''
    Método para probar hacer un completion genérico con un LLM
    '''

    # Crear el cliente de LLM
    llm_client = LLMClient()

    # Crear el prompt a user
    prompt = "Explica como funciona la API de Jira en Python."
    
    # Generar el texto
    response = llm_client.generate_text(prompt)
    
    # Imprimir la respuesta
    print("LLM Response:")
    print(response)


async def create_output_table_async(issues: List[IssueInfo]) -> None:
    '''
    Método que hace el procesamiento de la información.
    
    Recibe una lista de información de issues. Genera la cadena de consulta y salida.'''
    logger = Logger()
    logger.info("Comenzando a construir la tabla de salida en versión ASÍNCRONA...")

    # Obtener la instancia del OutputManager
    output_manager = OutputManager()

    # Crear el objeto de tipo OutputRunnable para la cadena
    output_runnable = OutputRunnable(output_manager)

    # Obtener parámetros de configuración
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("LLM_API_KEY")

    # Crear prompt desde una plantilla. Parametrizado a variables
    prompt = ChatPromptTemplate.from_template("""
    Eres un asistente que resume información de issues de Jira para reportes de negocio.

    Analiza los siguientes datos de un issue:
    - Clave del issue: {key}
    - Épica del issue: {epic_key}
    - Fecha de resolución: {resolution_date}                  
    - Resumen original: {summary}
    - Descripción: {description}
    - Documento de valor de negocio: {business_info}

    Genera una respuesta estructurada con los siguientes campos:

    1. "resumen": descripción breve (máximo 10 palabras) que explica de qué se trata el issue.
    2. "valor_negocio": resumen (máximo 25 palabras) del valor de negocio aportado por la HU, usando únicamente la sección de "objetivos de la iniciativa" del documento.
    3. "metrica_impactada": nombre de la métrica más impactada por la HU, sin explicaciones adicionales.
    4. "impactos_globales": el impacto que la HU tiene en todas las métricas definidas en la sección correspondiente, con nivel "Nulo", "Bajo", "Medio" o "Alto".
    5. "justificaciones": la justificación para cada uno de los impactos del punto anterior, con nombre de métrica y justificación.
    6. "issue_key": la clave de identificación del issue de Jira (por ejemplo, "SVA-1000").
    7. "epic_key": la clave de identificación de la épica a la que pertenece el issue (por ejemplo: GOBI-800).
    8. "resolution_date": la fecha en la que se resolvió el issue, expresada en formato MM-DD
    """)

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0
        )
    
    # Crear LLM con salida estructurada. Este paso es crítico.
    # Genera un RunnableBinding que hace un wrapper de LLM comportamiento adicional
    # (en este caso, la capacidad de manejar salida estructurada)
    structured_llm = llm.with_structured_output(IssueAnalysis)

    # La "cadena" de ejecución. De tipo RunnableSequence
    if not RATE_LIMITING:
        chain = prompt | structured_llm | output_runnable
    else:
        # Incorpora el limitador para esperar por cada llamada
        logger.info("Usaremos un limitador de llamadas para no exceder la tasa permitida...")
        chain = prompt | RateLimitingRunnable() | structured_llm | output_runnable

    # Introduciremos ejecución asincrónica
    inputs = [
        {
            "key": issue.key,
            "epic_key": issue.epic_key,
            "resolution_date": issue.resolution_date,
            "summary": issue.summary,
            "description": issue.description,
            "business_info": issue.business_info
        }
        for issue in issues        
    ]

    try:
        logger.info("Iniciando ejecución asíncrona del proceso...")
        results = await chain.abatch(inputs, max_concurrency=5)

    except Exception as e:
        logger.error(f"Error al procesar el issue: {e}")
    
    logger.info("Guardando archivo de salida...")
    output_manager.save_table_to_csv("output_table.csv")


def create_output_table(issues: List[IssueInfo]) -> None:
    '''
    Método que hace el procesamiento de la información.
    
    Recibe una lista de información de issues. Genera la cadena de consulta y salida.'''
    logger = Logger()
    logger.info("Comenzando a construir la tabla de salida en versión SÍNCRONA...")

    # Obtener la instancia del OutputManager
    output_manager = OutputManager()

    # Crear el objeto de tipo OutputRunnable para la cadena
    output_runnable = OutputRunnable(output_manager)

    # Obtener parámetros de configuración
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("LLM_API_KEY")

    # Crear prompt desde una plantilla. Parametrizado a variables
    prompt = ChatPromptTemplate.from_template("""
    Eres un asistente que resume información de issues de Jira para reportes de negocio.

    Analiza los siguientes datos de un issue:
    - Clave del issue: {key}
    - Épica del issue: {epic_key}
    - Fecha de resolución: {resolution_date}                  
    - Resumen original: {summary}
    - Descripción: {description}
    - Documento de valor de negocio: {business_info}

    Genera una respuesta estructurada con los siguientes campos:

    1. "resumen": descripción breve (máximo 10 palabras) que explica de qué se trata el issue.
    2. "valor_negocio": resumen (máximo 25 palabras) del valor de negocio aportado por la HU, usando únicamente la sección de "objetivos de la iniciativa" del documento.
    3. "metrica_impactada": nombre de la métrica más impactada por la HU, sin explicaciones adicionales.
    4. "impactos_globales": el impacto que la HU tiene en todas las métricas definidas en la sección correspondiente, con nivel "Nulo", "Bajo", "Medio" o "Alto".
    5. "justificaciones": la justificación para cada uno de los impactos del punto anterior, con nombre de métrica y justificación.
    6. "issue_key": la clave de identificación del issue de Jira (por ejemplo, "SVA-1000").
    7. "epic_key": la clave de identificación de la épica a la que pertenece el issue (por ejemplo: GOBI-800).
    8. "resolution_date": la fecha en la que se resolvió el issue, expresada en formato MM-DD
    """)

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0
        )
    
    # Crear LLM con salida estructurada. Este paso es crítico.
    # Genera un RunnableBinding que hace un wrapper de LLM comportamiento adicional
    # (en este caso, la capacidad de manejar salida estructurada)
    structured_llm = llm.with_structured_output(IssueAnalysis)

    # La "cadena" de ejecución. De tipo RunnableSequence
    chain = prompt | structured_llm | output_runnable

    for issue in issues:
        logger.info(f"Procesando issue {issue.key} para tabla de salida...")

        # Crear la estructura de entrada, con los parámetros que espera el prompt
        issue_data = {
            "key": issue.key,
            "epic_key": issue.epic_key,
            "resolution_date": issue.resolution_date,
            "summary": issue.summary,
            "description": issue.description,
            "business_info": issue.business_info
        }

        try:
            # Invocar la cadena. Esto genera la ejecución del RunnableSequence. En este caso,
            # el prompt que entra en el LLM con estructura
            logger.info("Ejecutando cadena de consulta...")
            result = chain.invoke(issue_data)
            if result:
                logger.info(f"Completado el ciclo para HU: {issue.key}...")


        except Exception as e:
            logger.error(f"Error al procesar el issue {issue.key}: {e}")
            continue
    
    logger.info("Guardando archivo de salida...")
    output_manager.save_table_to_csv("output_table.csv")



if __name__ == "__main__":
    main()