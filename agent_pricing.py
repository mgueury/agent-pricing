from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from typing import List, Tuple, Dict, Any
from langchain.tools import Tool
import os
import re
from typing import TypedDict, Literal
import oracledb
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langgraph.prebuilt import create_react_agent
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from diagrams.oci.compute import VM
from diagrams.oci.database import Stream
from diagrams import Diagram, Cluster
from diagrams.oci import connectivity, network

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    messages: List[Tuple[str, str]]

DB_CONFIG = {
    "user": "admin",
    "password": os.getenv("TF_VAR_db_password"),
    "config_dir": "wallet",
    "wallet_location": "wallet",
    "wallet_password": os.getenv("TF_VAR_db_password"),
    "thick_mode": "False"
}

connection = oracledb.connect(
    config_dir=DB_CONFIG["config_dir"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
    dsn=DB_CONFIG["dsn"],
    wallet_location=DB_CONFIG["wallet_location"],
    wallet_password=DB_CONFIG["wallet_password"]
)

embedding =  OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=os.getenv("TF_VAR_compartment_ocid"))

load_dotenv()
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

@tool
def get_oci_cost(service_type: str) -> str:
    """
    Retrieves the cost of OCI services (Compute memory & OCPU) from the database.
    Args:
        service_type (str): The type of OCI service ('CPU' or 'Memory').

    Returns:
        str: A string containing the cost of the specified OCI service, or an error message.
    """
    print("Service Type: ", service_type)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT product_name, unit_cost, currency, description FROM oci_products WHERE product_type = :1", [service_type])

            return [{
                "product_name": row[0],
                "unit_cost": row[1],
                "currency": row[2],
                "description": row[3]
            } for row in cursor]
    except Exception as e:
        return f"Error retrieving OCI cost: {e}"

@tool
def get_building_block_cost(block_type: str) -> str:
    """Retrieves the cost of media streaming building blocks (ingest, transcode, distribution).

    Args:
        block_type (str): The type of building block ('ingest', 'transcode', 'distribution').

    Returns:
        str: A string containing the cost of the specified building block options, or an error message.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT product_name, unit_cost, currency, description FROM gc_products WHERE product_type = :1", [block_type])

            return [{
                "product": row[0],
                "unit_cost": row[1],
                "currency": row[2],
                "description": row[3]
            } for row in cursor]

    except Exception as e:
        return f"Error retrieving building block cost: {e}"

@tool
def rag_search(query: str) -> str:
    """Performs a RAG (Retrieval-Augmented Generation) search on the PDF document.

    Args:
        query (str): The search string to use for retrieving information from the PDF.

    Returns:
        str: Relevant content from the PDF based on the search query, or a message if no results are found.
    """
    try:

        vector_store = OracleVS(
            client=connection,
            embedding_function=embedding,
            table_name="gc_rag",
            distance_strategy=DistanceStrategy.COSINE,
        )

        docs = vector_store.similarity_search(query, 1)

        if docs:
            # Extract and concatenate the content from the retrieved documents
            results = "\n".join([doc.page_content for doc in docs])
            return results
        else:
            return "No relevant information found in the PDF."
    except Exception as e:
        return f"Error during RAG search: {e}"

import json

count = 1
@tool
def generate_architecture_diagram(components: str) -> str:
    """
    Generates an architecture diagram based on the specified components.
    The input should be a JSON string with the following structure:

    {
       "ingest": "Label for Ingest",
       "transcode": ["Label for Transcode 1", "Label for Transcode 2", ...],
       "distribution": ["Label for Distribution 1", "Label for Distribution 2", ...]
    }

    The diagram will show:
      - An Ingest component labeled with the 'ingest' value.
      - Each Transcode component (from the list) connected from the ingest node.
      - Each Distribution component (from the list) connected from each transcode node.

    Returns:
        str: A markdown snippet with the architecture diagram image embedded.
    """
    print("GENERATING ARCHITECTURE DIAGRAM")
    global count
    try:
        comp = json.loads(components)
    except Exception as e:
        return f"Error parsing JSON: {e}"
    print("COMPONENTS: ", comp)
    # Validate required keys.
    if not all(key in comp for key in ["ingest", "transcode", "distribution"]):
        return "Error: JSON must contain 'ingest', 'transcode', and 'distribution' keys."

    ingest_label = comp["ingest"]
    transcode_list = comp["transcode"]
    distribution_list = comp["distribution"]

    if not isinstance(transcode_list, list) or not isinstance(distribution_list, list):
        return "Error: 'transcode' and 'distribution' must be lists."

    static_dir = "./"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    filename_only = f"architecture_diagram{count}"
    diagram_file = os.path.join(static_dir, filename_only)

    with Diagram("Media Streaming Architecture", filename=diagram_file, show=False, outformat="png"):
        ingest_node = Stream(ingest_label)
        if len(transcode_list) > 1:
            with Cluster("Transcodings"):
                transcode_nodes = [VM(label) for label in transcode_list]
        else:
            transcode_nodes = [VM(label) for label in transcode_list]

        if len(distribution_list) > 1:
            with Cluster("Distribution"):
                distribution_nodes = []
                for label in distribution_list:
                    if "cdn" in label.lower():
                        max_label_length = 18 
                        if len(label) > max_label_length:
                            label = label[:max_label_length] + "..." 
                        node = connectivity.CDN(label)
                    else:
                        node = network.LoadBalancer(label)
                    distribution_nodes.append(node)
        else:
            distribution_nodes = []
            for label in distribution_list:
                if "cdn" in label.lower():
                    max_label_length = 200 
                    if len(label) > max_label_length:
                        label = label[:max_label_length] + "..." 
                    node = connectivity.CDN(label)
                else:
                    node = network.LoadBalancer(label)
                distribution_nodes.append(node)

        for t_node in transcode_nodes:
            ingest_node >> t_node
            for d_node in distribution_nodes:
                t_node >> d_node

    backend_url = os.environ.get("BACKEND_URL") or "http://127.0.0.1:8000"
    diagram_url = f"{backend_url}/static/{filename_only}.png"
    count += 1

    return f"Architecture diagram generated successfully: {filename_only}.png\n\n![Architecture Diagram]({diagram_url})"

tools = [
    Tool(
        name="get_oci_cost",
        func=get_oci_cost,
        description="Useful for retrieving the cost of OCI services (Compute memory & OCPU). Input should be 'CPU' or 'Memory'."
    ),
    Tool(
        name="get_building_block_cost",
        func=get_building_block_cost,
        description="Useful for retrieving the cost of media streaming building blocks (ingest, transcode, distribution). Input should be one of these options."
    ),
    Tool(
        name="rag_search",
        func=rag_search,
        description="Useful for searching the media streaming guidelines PDF. Input should be a search query related to transcoding, resource requirements, or best practices."
    ),
    Tool(
        name="generate_architecture_diagram",
        func=generate_architecture_diagram,
        description="Generates an architecture diagram based on the specified components. Input should be a comma-separated string of components (e.g., 'ingest,transcode,cdn')."
    )
]


model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
# model = ChatOCIGenAI(model_id="c",
#                       service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
#                       compartment_id="ocid1.compartment.oc1..aaaaaaaa6gapyx7754dtzpcq3x6h5fmdqjboryka6e2vndc7uds5pmqsqvuq",
#                       model_kwargs={"temperature": 0.7, "max_tokens": 4000})

tool_names = [tool.name for tool in tools]

prompt = PromptTemplate(template="""
Introduction and User Engagement:

Greet the user warmly and introduce yourself as an AI assistant designed to assist in configuring media streaming setups.
Clearly outline the information required to provide accurate recommendations and pricing.
This should be a conversation where you ask for information piece by piece to provide a simple and pleasant user experience.
Ensure the output is clear, concise, easy to understand and well formatted.
If you are given all of the information in one go, you can skip the questions and directly provide the final output, which must always include an architecture diagram by calling the 'generate_architecture_diagram' tool.

Existing OCI Infrastructure:

Ask the user if they have an existing Oracle Cloud Infrastructure (OCI) setup.

If yes, confirm the current specifications (e.g., number of OCPUs and amount of RAM).

If no, inform them that the necessary OCI resources will be provisioned by default.

Transcoding Requirements:

Ask if the user requires transcoding services for their media streams.

If yes, use the appropriate tool to fetch all available transcoding options (including details such as resolution, bitrate, and cost).
                        
Always show the user all 3 transcoding options, if the user provides details on the content type, recommend the best option.

Use a RAG search tool to consult guidance documents and recommend the best transcoding option based on the content type (e.g., sports, mobile streaming).

If the user selects multiple transcoding, ensure that you sum the OCPU and RAM requirements for all selected options. For example, 25Mbps 1080p requires 4 OCPUs and 32 GB RAM, 15Mbps 720p requires 2 OCPUs and 16 GB RAM - If the user selects both and has 1 OCPU and 8GB RAM, they will need an additional 5 OCPUs and 40 GB RAM.

Distribution Method:

Ask for the preferred distribution method(s): Content Delivery Network (CDN), Direct IP, or Both.

If Direct IP is chosen, ask for the number of endpoints and their IPs or FQDNs.

Use the relevant tool to fetch cost details for each distribution method.

Operational Duration:

Ask for the expected duration (in hours) the setup will run.

Resource Recommendation and Validation:

Based on the chosen transcoding option, use a RAG search to consult guidance on recommended OCI resources (OCPUs and Memory).

If the current OCI setup does not meet the requirements:

Recommend the necessary upgrades.

Use the OCI cost tool to fetch current pricing for additional resources.

Present these upgrade costs and ask for confirmation.

Pricing Calculation and Architecture Design:

Calculate the total cost based on transcoding, distribution, any required OCI resource upgrades and number of hours. Produce the architecture diagram.
The cost should always include the transcoding cost for the selected options, and the distribution cost. If more OCI resources are needed, include those costs as well.

Provide a detailed breakdown of costs for transparency.

Architecture Diagram Generation:

MANDATORY STEP: You MUST call the tool generate_architecture_diagram with a JSON object containing the keys "ingest", "transcode", and "distribution".

For example, if the ingest is "Live Sports Feed", the transcoding option is ["15Mbps 720p"], and the distribution is ["CDN - Global"], then call the tool with:
{{ "ingest": "Live Sports Feed", "transcode": ["15Mbps 720p"], "distribution": ["CDN - Global"] }}

IMPORTANT: Do NOT output the raw JSON. Instead, capture ONLY the markdown snippet (which embeds the diagram image) returned by the tool.

This tool call MUST be performed and its result included exactly once in your final output.

Final Confirmation and Next Steps:

Present a single, consolidated final answer that includes:

A summary of the recommended configuration, pricing details and any required OCI resource upgrades.

The single embedded architecture diagram (insert the markdown snippet returned by the generate_architecture_diagram tool here).
                        
You must execute the tool to create the diagram.

Ensure the final output is clear, concise, and free of repeated sections or raw tool outputs.

Ask the user if they would like to proceed or if adjustments are needed.
                        
Ingest, Transcode and Distribution costs can be found by calling the 'get_building_block_cost' tool.

Additional Requirements:
When showing the multiplication calculations, do not say '\times', instead use 'x' (e.g., 4 x 5 = 20).
When printing a table do use '<br>' in the Description.
                        
You have access to the following tools:
{tools}
""",
input_variables=["tools", "tool_names"])

prompt_text = prompt.format(tools=tools, tool_names=tool_names)

memory = MemorySaver()

graph = create_react_agent(model, tools=tools, prompt=prompt_text, checkpointer=memory)
#graph = create_react_agent(model, tools=tools, prompt=prompt_text, checkpointer=None)

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            #message.pretty_print()
            print(message)

inputs = {"messages": [("user", f"I'd like to get a price to ingest a live sports stream for 4 hours. Could you please suggest a setup and provide me with the end cost? Don't ask me any questions, just make assumptions and tell me the assumptions you've made.")]}
#print_stream(graph.stream(inputs,{"thread_id": "1"} ,stream_mode="values")) # Comment out this line to prevent printing in console during FastAPI execution

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessage(BaseModel):
    message: str

import re

def deduplicate_text(text: str) -> str:
    paragraphs = re.split(r'\n\s*\n', text)
    seen = set()
    unique_paragraphs = []
    for para in paragraphs:
        trimmed = para.strip()
        if trimmed and trimmed not in seen:
            seen.add(trimmed)
            unique_paragraphs.append(trimmed)
    return "\n\n".join(unique_paragraphs)

def chat(prompt):
    inputs = {"messages": [("user", prompt)]}
    response_stream = graph.stream(inputs, {"thread_id": "1"}, stream_mode="values")
    response = ""
    print("--- RAW RESPONSE STREAM START ---") # DEBUG LOG
    for s in response_stream:
        print("STREAM CHUNK:", s) # DEBUG LOG
        message = s["messages"][-1] # Get the LAST message in the list, which should be the latest response
        print("MESSAGE OBJECT:", message) # DEBUG LOG
        if isinstance(message, AIMessage): # Check if it's an AIMessage object
            print("AIMessage OBJECT FOUND:", message.content) # DEBUG LOG
            response += message.content # Access content attribute
        elif isinstance(message, HumanMessage): # Optionally handle HumanMessage if needed, though unlikely in agent response stream
            print("HumanMessage OBJECT FOUND:", message.content) # DEBUG LOG - for completeness
            response += "" # Or handle HumanMessage content if relevant
        elif isinstance(message, tuple) and message[0] == "assistant": # Keep this for any potential tuple-based messages
            print("TUPLE ASSISTANT MESSAGE FOUND:", message[1]) # DEBUG LOG
            response += message[1]
        elif isinstance(message, str): # Keep this for any potential string messages
            print("STRING MESSAGE FOUND:", message) # DEBUG LOG
            response += message
        else:
            print("UNKNOWN MESSAGE TYPE:", message) # DEBUG LOG - for unexpected types

    print("--- RAW RESPONSE STREAM END ---") # DEBUG LOG


    # Remove echoed user input if present.
    user_text = user_message.message.strip()
    trimmed_response = response.strip()
    if trimmed_response.startswith(user_text):
        response = trimmed_response[len(user_text):].lstrip()

    # Remove any unwanted JSON blocks or raw tool outputs.
    response = re.sub(r'^\s*\[[^\]]*\]', '', response).strip()
    response = re.sub(
        r'Media Streaming Setup Guidelines Transcoding Guidelines[\s\S]*?(?=For live sports streaming)',
        '',
        response
    ).strip()

    # Deduplicate repeated paragraphs.
    clean_response = deduplicate_text(response)

    print("FINAL RESPONSE TO STREAMLIT:", clean_response) # DEBUG LOG
    return {"response": clean_response}
