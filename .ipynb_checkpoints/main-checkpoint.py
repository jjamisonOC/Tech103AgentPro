from agentpro import AgentPro
from agentpro.tools import AresInternetTool, CodeEngine, YouTubeSearchTool, SlideGenerationTool
import os
import dotenv

def main():
    # Load environment variables from .env file if it exists
    dotenv.load_dotenv()
    
    # Check for required API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the agent.")
        return
        
    if not os.environ.get("TRAVERSAAL_ARES_API_KEY"):
        print("Warning: TRAVERSAAL_ARES_API_KEY environment variable is not set.")
        print("AresInternetTool will not be available.")
        tools = [CodeEngine(), YouTubeSearchTool(), SlideGenerationTool()]
    else:
        tools = [
            AresInternetTool(),
            CodeEngine(), 
            YouTubeSearchTool(), 
            SlideGenerationTool()
        ]
    
    # Create agent with tools
    agent = AgentPro(tools=tools)
    
    print("AgentPro is initialized and ready. Enter 'quit' to exit.")
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
            
        try:
            response = agent(user_input)
            print(f"\nAgent Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()