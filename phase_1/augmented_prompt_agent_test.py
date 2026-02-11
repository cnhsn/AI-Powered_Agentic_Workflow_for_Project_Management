import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "a helpful travel guide specializing in European destinations"
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

prompt = "What is the capital of France?"
augmented_agent_response = augmented_agent.respond(prompt)

print(f"Persona: {persona}")
print(f"Prompt: {prompt}")
print(f"Response: {augmented_agent_response}")
