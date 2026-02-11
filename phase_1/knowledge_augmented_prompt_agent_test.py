# Test script for KnowledgeAugmentedPromptAgent
# Demonstrates agent using provided knowledge instead of general LLM knowledge

import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Provide intentionally incorrect knowledge to test knowledge usage
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

prompt = "What is the capital of France?"
response = knowledge_agent.respond(prompt)

print(f"Persona: {persona}")
print(f"Knowledge: {knowledge}")
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\nThe agent's response explicitly uses the provided knowledge rather than its inherent LLM knowledge.")
