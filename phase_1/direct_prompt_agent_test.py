import os
from dotenv import load_dotenv
from workflow_agents.base_agents import DirectPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

direct_agent = DirectPromptAgent(openai_api_key)

prompt = "What is the Capital of France?"
response = direct_agent.respond(prompt)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\nKnowledge Source: The agent uses general knowledge from the gpt-3.5-turbo LLM model.")
