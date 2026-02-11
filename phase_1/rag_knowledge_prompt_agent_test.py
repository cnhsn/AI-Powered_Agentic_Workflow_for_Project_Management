import os
from dotenv import load_dotenv
from workflow_agents.base_agents import RAGKnowledgePromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "a knowledgeable geography expert"
knowledge_documents = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "London is the capital of the United Kingdom.",
    "Berlin is the capital of Germany.",
    "Rome is the capital of Italy."
]

rag_agent = RAGKnowledgePromptAgent(openai_api_key, persona, knowledge_documents)

prompt = "What is the capital of France?"
response = rag_agent.respond(prompt)

print(f"Persona: {persona}")
print(f"Prompt: {prompt}")
print(f"Response: {response}")
