from adk import Agent, AgentClient
from adk.tool.prebuilt import AgenticRagTool
from custom_function_tools import AccountToolkit

"""
This example shows how an agent with multiple tools work, and how multi-turn runs work.
"""

def main():

    # Assuming the resources were already provisioned
    agent_endpoint_id = "ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaacqy6p4qaxlkvovo4kmtwqn3j3tirawd7g23t5rzmxmqfiflza3iq"
    knowledge_base_id = "ocid1.genaiagentknowledgebaseppe.oc1.us-chicago-1.amaaaaaacqy6p4qayncpcwp4gcvc7tokfzcpjglwme3qwf4iklpnjmkjif3q"

    client = AgentClient(
        auth_type="security_token",
        profile="BoatOc1",
        runtime_endpoint="https://ppe.agent-runtime.generativeai.us-chicago-1.oci.oraclecloud.com",
        management_endpoint="https://ppe-agent.generativeai.us-chicago-1.oci.oraclecloud.com",
    )

    instructions = """
    You are customer support agent.
    Use KB tool to answer product questions.
    Use tools to fetch user and org info by id.
    Only orgs of Enterprise plan can use Responses API.
    """

    agent = Agent(
        client=client,
        agent_endpoint_id=agent_endpoint_id,
        instructions=instructions,
        tools=[AgenticRagTool(knowledge_base_ids=[knowledge_base_id]), AccountToolkit()]
    )

    agent.setup()

    # This is a context your existing code is best at producing (e.g., fetching the authenticated user id)
    client_provided_context = "[Context: The logged in user ID is: user_123] "

    # Handle the first user turn of the conversation
    input = "What is the Responses API?"
    input = client_provided_context + " " + input
    response = agent.run(input)
    response.pretty_print()

    # Handle the second user turn of the conversation
    input = "Is my user account eligible for the Responses API?"
    input = client_provided_context + " " + input
    response = agent.run(input, session_id=response.session_id)
    response.pretty_print()


if __name__ == "__main__":
    main()
