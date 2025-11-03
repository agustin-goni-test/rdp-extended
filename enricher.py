import operator
from typing import Annotated, Dict, List, TypedDict, Optional, Any
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent # No AgentExecutor present. Is that a problem?
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from dotenv import load_dotenv
from jira_client import JiraClient

load_dotenv()

jira_client = JiraClient()


# Define the state
class EnrichmentState(TypedDict):
    original_text: str
    current_text: str
    detected_conditions: List[str]
    tool_inputs: Dict[str, str]
    tool_results: Dict[str, Dict]
    enrichment_steps: List[Dict]
    iteration: int
    max_iterations: int
    should_continue: bool

@tool
def obtain_user_id(usernames: List[str]) -> Dict[str, str]:
    '''Tool to replace user names for user ids.'''
    results = jira_client.get_users_id(usernames)

    # Convert to the expected format: {username: user_id}
    user_id_map = {}
    for result in results:
        # Assuming result is a dict with 'name' and 'id' keys
        user_id_map[result['query_name']] = result['id']
    
    return user_id_map


class JQLEnrichmentAgent:
    def __init__ (self, llm, jira_client, max_iterations: int = 3):
        self.llm = llm
        self.jira_client = jira_client
        self.max_iterations = max_iterations

        # Define tools
        self.tools = [obtain_user_id]
        self.tool_node = ToolNode(self.tools)

        # Define detection promt to raise conditions
        self.detection_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", '''
                You are a JQL analysis expert. You're job is to analyze an input text, which is a JQL expression, and find if some enrichment
            conditions are present.
             
            Condition of enrichment you need to detect:
             1. user_lookup - When usernames are present in assignees, reporters, watcher, or other fields related to Jira users. In this condition,
             you are looking for names of a person (e.g, "Edgar Benitez") not IDs (e.g, "634d9d8bfc0cc7a600ac3d53")
             2. epic_lookup - When epic names are referenced. In this condition you are searching for names, such as "Anticipo KLAP", not codes, such as "GOBI-895"

             For each enrichment needed, extract the specific values that need to be reprocessed. Return a JSON with this structured:
             {{
                "enrichments_needed": ["user_lookup", "epic_lookup"],
                "extracted_data": {{
                    "user_lookup": ["username1", "username2"],
                    "epic_lookup": ["epicname1", "epicname2"]
                }}
             }}            
             '''),
             ("human", "JQL: {text}")
             ])
        
        # self.input_extraction_prompt = ChatPromptTemplate.from_messages([
        #     ("system", '''
        #         Extract the specific parts of the input expression that are relevant for the {tool_name} tool. Focus only on the
        #      portion you need for processing by this specific tool. Return only the necessary information, without additional comments.
        #      '''),
        #      ("human", "JQL expression: {text}")
        # ])

        # self.input_extraction_prompt = ChatPromptTemplate.from_messages([
        #     ("system", '''
        #         Extract the names of the users included in the input JQL expression
             
        #      Return a JSON array of the names and nothing more. Name that array "names_included".
        #      '''),
        #      ("human", "JQL expression: {text}")
        # ])

        # Build the graph
        self.graph = self._build_graph()


    def _build_graph(self) -> StateGraph:
        '''Build the graph needed for the execution'''
        workflow = StateGraph(EnrichmentState)

        # Add the nodes
        workflow.add_node("detect_and_extract", self.detect_and_extract)
        workflow.add_node("execute_tools", self.execute_tools)
        workflow.add_node("merge_results", self.merge_results)
        workflow.add_node("check_continuation", self.check_continuation)

        workflow.set_entry_point("detect_and_extract")
        workflow.add_edge("detect_and_extract", "execute_tools")
        workflow.add_edge("execute_tools", "merge_results")
        workflow.add_edge("merge_results", "check_continuation")
        workflow.add_conditional_edges(
            "check_continuation",
            self.should_continue_logic,
            {"continue": "detect_and_extract", "end": END}
        )

        return workflow.compile()



    def detect_and_extract(self, state:EnrichmentState) -> EnrichmentState:
        '''Do detection of condition and extraction at the same time'''
        text = state["current_text"]

        chain = self.detection_extraction_prompt | self.llm
        response = chain.invoke({"text": text})

        try:
            # Parse the combined result
            content = response.content

            # Clean up the JSON response if it has extra characters
            if '```json' in content:
                start_index = content.find('```json') + 7
                end_index = content.find('```', start_index)
                json_str = content[start_index:end_index].strip()

            elif '```' in content:
                start_index = content.find('```') + 3
                end_index = content.find('```', start_index)
                json_str = content[start_index:end_index].strip()

            else:
                json_str = content.strip()

            data = json.loads(json_str)
            state["detected_conditions"] = data["enrichments_needed"]
            state["tool_inputs"] = data["extracted_data"]


        except json.JSONDecodeError:
            print(f"JSON parsing failed. Content was: {response.content}")
            state["detected_conditions"] = []
            state["tool_inputs"] = {}

        return state
    
    def execute_tools(self, state: EnrichmentState) -> EnrichmentState:
        '''Execute the tools with the extracted data'''
        tool_inputs = state["tool_inputs"]
        tool_results = {}

        # Handle the case for each tool
        if "user_lookup" in tool_inputs:
            # Obtain the list of names
            usernames = tool_inputs["user_lookup"]
            results = obtain_user_id.invoke({"usernames": usernames})
            tool_results["user_lookup"] = results

        if "epic_lookup" in tool_results:
            pass

        state["tool_results"] = tool_results
        return state
    

    def merge_results(self, state: EnrichmentState) -> EnrichmentState:
        '''Incorporate the tool results'''
        text = state["current_text"]
        tool_results = state["tool_results"]
        detected_conditions = state["detected_conditions"].copy()  # Use a copy to make changes

        if "user_lookup" in tool_results:
            user_id_map = tool_results["user_lookup"]

            # Replace name with id, reagarless of whether quotation marks are present
            for username, user_id in user_id_map.items():
                # Replace quoted version
                text = text.replace(f'"{username}"', user_id)
                # Replace unquoted version
                text = text.replace(username, user_id)
            
            # If this was one of the conditions detected and the id map exists
            # Then this condition will be deleted to avoid iterations
            if "user_lookup" in detected_conditions and user_id_map:
                detected_conditions.remove("user_lookup")


        state["current_text"] = text
        state["detected_conditions"] = detected_conditions  # Update with whatever was eliminated
        return state
    

    def should_continue_logic(self, state: EnrichmentState) -> EnrichmentState:
        '''Determine if the workflow must continue'''
        return "continue" if state.get("should_continue", False) else "end"
    

    def enrich(self, jql_text: str) -> str:
        """Main method to enrich JQL"""
        initial_state = {
            "original_text": jql_text,
            "current_text": jql_text,
            "detected_conditions": [],
            "tool_inputs": {},
            "tool_results": {},
            "enrichment_steps": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "should_continue": True
        }

        final_state = self.graph.invoke(initial_state)

        return final_state["current_text"]

        

    def check_continuation(self, state: EnrichmentState) -> EnrichmentState:
        """Simple version - usually just one iteration for JQL enrichment"""
        iteration = state.get("iteration", 0) + 1
        state["iteration"] = iteration

        # Consider two condition to stop
        # No tools needed and max number of iterations exceeded
        conditions_detected = len(state.get("detected_conditions", [])) > 0
        under_max_iterations = iteration < self.max_iterations

        state["should_continue"] = conditions_detected and under_max_iterations
        return state





    def detect_conditions(self, state: EnrichmentState) -> EnrichmentState:
        '''Method to detect conditions in the expression'''
        text = state["current_text"]

        # Use LLM to detect conditions
        chain = self.detection_prompt | self.llm
        response = chain.invoke({"text": text})

        try:
            # Parse JSON response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)

            # Extract JSON from response
            start_index = content.find('{')
            end_index = content.rfind('}') + 1
            if start_index != -1 and end_index != -1:
                json_str = content[start_index:end_index]
                condition_data = json.loads(json_str)
                detected_conditions = list(condition_data.keys())
            else:
                detected_conditions = []

        except json.JSONDecodeError:
            detected_conditions = self._fallback_condition_detection(text)
        
        state["detected_conditions"] = detected_conditions
        return state
    

    def _fallback_condition_detection(self, text: str) -> List[str]:
        '''Possile implementation of a "fallback" method'''
        return []


    def extract_tool_inputs(self, state: EnrichmentState) -> EnrichmentState:
        '''Extract specific input for each tool'''

        detected_conditions = state["detected_conditions"]
        text = state["current_text"]
        tool_inputs = {}

        conditions_to_tools = {
            "user_name_included": "user"
        }
        
        






