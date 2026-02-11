# Test script for AugmentedPromptAgent
# Demonstrates how persona shapes the tone and style of responses

import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Define persona to shape response style
persona = "a helpful travel guide specializing in European destinations"
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# Test with same question to see persona's effect on response
prompt = "What is the capital of France?"
augmented_agent_response = augmented_agent.respond(prompt)

print(f"Persona: {persona}")
print(f"Prompt: {prompt}")
print(f"Response: {augmented_agent_response}")
print()
print("Knowledge Source: The AugmentedPromptAgent still uses the LLM's general knowledge to answer questions.")
print("Persona Impact: The persona is used to shape the tone and style of the response, making it more specialized and contextually appropriate for the given role.")
