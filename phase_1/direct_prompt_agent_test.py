# Test script for DirectPromptAgent
# Demonstrates basic agent that uses only LLM's general knowledge

import os
from dotenv import load_dotenv
from workflow_agents.base_agents import DirectPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Create agent instance
direct_agent = DirectPromptAgent(openai_api_key)

# Test with a simple factual question
prompt = "What is the Capital of France?"
response = direct_agent.respond(prompt)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\nKnowledge Source: The agent uses general knowledge from the gpt-3.5-turbo LLM model.")
