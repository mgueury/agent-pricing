from rich.console import Console
from adk import Agent, AgentClient
from adk import Toolkit, tool
from typing import Dict, Any
import os

import os
import streamlit as st

console = Console(log_time=False, log_path=False)

class SearchToolkit(Toolkit):

    @tool
    def search(self, topic: str) -> Dict[str, Any]:
        """ Search for a given topic.

        Args:
            topic (str): The topic to get info for

        Returns:
            str: A JSON string containing the search result
        """
        ## need to set ("TAVILY_API_KEY")
        tavily_search = TavilySearchResults(max_results=3)
        search_docs = tavily_search.invoke(question)
        return search_docs

    '''
    [{'url': 'https://www.datacamp.com/tutorial/langgraph-tutorial',
    'content': 'LangGraph is a library within the LangChain ecosystem designed to tackle these challenges head-on. LangGraph provides a framework for defining, coordinating, and executing multiple LLM agents (or chains) in a structured manner.'},
    {'url': 'https://langchain-ai.github.io/langgraph/',
    'content': 'Overview LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures, differentiating it from DAG-based solutions. As a ...'},
    {'url': 'https://www.youtube.com/watch?v=nmDFSVRnr4Q',
    'content': 'LangGraph is an extension of LangChain enabling Multi-Agent conversation and cyclic chains. This video explains the basics of LangGraph and codesLangChain in...'}]      
    '''

SETUP=False



def setup():

    # Assuming the agent and agent endpoint resources were already provisioned
    adk_demo_agent_endpoint_id = os.getenv("TF_VAR_agent_endpoint_ocid")

    # Initialize the agent client
    client = AgentClient(
        auth_type="instance_principal",
        region="fra"
    )

    # Instantiate the local agent object (with the client, instructions, and tools to be registered)
    global agent
    agent = Agent(
        agent_endpoint_id=adk_demo_agent_endpoint_id,
        client=client,
        instructions="""You are a helpful assistant that can perform search
        And create result document in this format:
        
        ## Title
        Summary of the content

        ## Details
        Response to the search question in 5 lines.
        """
        tools=[SearchToolkit()]
    )

    # Set up the agent once (this configures the instructions and tools in the remote agent resource)
    if SETUP:
        agent.setup()


def demo():

    # Use the agent to process the end user request (it automatically handles the function calling)
    input = "What is the square root of 475695037565?"
    response = agent.run(input, max_steps=3)
    response.pretty_print()

    # second turn
    input = "do the same thing for 123456789"
    response = agent.run(input, session_id=response.session_id, max_steps=3)
    response.pretty_print()

st.write("# Welcome to ADK Assistant!")
st.write("Greetings!")
setup()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# Get new message from the user
prompt = st.chat_input("Type your message here...")

if prompt:
    # Display the user's message
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the FastAPI langgraph agent at the /chat/ endpoint.
    try:
        # res = agent_pricing.chat(prompt)
        res = agent.run(prompt, max_steps=3)

        response_data = res.json()
        assistant_response = response_data.get("response", "No response received.")
    except Exception as e:
        assistant_response = f"Error calling agent: {e}"

    # Display the assistant's response and update chat history
    st.session_state.chat_history.append(("assistant", assistant_response))
    with st.chat_message("assistant"):
        st.markdown(assistant_response)    