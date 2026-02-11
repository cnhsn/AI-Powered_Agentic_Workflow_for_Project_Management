import os
from dotenv import load_dotenv
from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"

worker_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

evaluation_persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should correctly state that Paris is the capital of France, not London"

evaluation_agent = EvaluationAgent(openai_api_key, evaluation_persona, evaluation_criteria, worker_agent, max_interactions=10)

prompt = "What is the capital of France?"
result = evaluation_agent.evaluate(prompt)

print(f"Prompt: {prompt}")
print(f"\nFinal Response: {result['final_response']}")
print(f"\nEvaluation: {result['evaluation']}")
print(f"\nIterations: {result['iterations']}")
