from rich.console import Console

from adk import Agent, AgentClient
from adk.tool.prebuilt import CalculatorToolkit


console = Console(log_time=False, log_path=False)


def main():

    try:
        # Initialize the agent client
        client = AgentClient(
            auth_type="security_token",
            profile="BoatOc1",
            region="us-chicago-1"
        )

        # Instantiate the local agent object (with the client, instructions, and tools to be registered)
        agent = Agent(
            client=client,
            agent_endpoint_id="ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaacqy6p4qaqzqhtgbj7arnj2rzy645g7tkeyqr2eaq27x4wfq3olsa",
            instructions="You are a helpful assistant that can perform calculations.",
            tools=[CalculatorToolkit()]
        )

        # Set up the agent once (this configures the instructions and tools in the remote agent resource)
        agent.setup()

        # Use the agent to process the end user request (it automatically handles the function calling)
        input = "What is the square root of 475695037565?"
        response = agent.run(input, max_steps=3)
        response.pretty_print()

    except Exception as e:
        console.print_exception()

if __name__ == "__main__":
    main()