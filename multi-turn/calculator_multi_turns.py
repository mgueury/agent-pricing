from rich.console import Console

from adk import Agent, AgentClient
from adk.tool.prebuilt import CalculatorToolkit
import os

console = Console(log_time=False, log_path=False)


def main():

    # Assuming the agent and agent endpoint resources were already provisioned
    adk_demo_agent_endpoint_id = os.getenv("TF_VAR_agent_endpoint_ocid")

    try:
        # Initialize the agent client
        client = AgentClient(
            auth_type="security_token",
            profile="BoatOc1",
            region="eu-frankfurt-01"
        )

        # Instantiate the local agent object (with the client, instructions, and tools to be registered)
        agent = Agent(
            agent_endpoint_id=adk_demo_agent_endpoint_id,
            client=client,
            instructions="You are a helpful assistant that can perform calculations.",
            tools=[CalculatorToolkit()]
        )

        # Set up the agent once (this configures the instructions and tools in the remote agent resource)
        agent.setup()

        # Use the agent to process the end user request (it automatically handles the function calling)
        input = "What is the square root of 475695037565?"
        response = agent.run(input, max_steps=3)
        response.pretty_print()

        # second turn
        input = "do the same thing for 123456789"
        response = agent.run(input, session_id=response.session_id, max_steps=3)
        response.pretty_print()

    except Exception as e:
        console.print_exception()

if __name__ == "__main__":
    main()