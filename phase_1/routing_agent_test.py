import os
from dotenv import load_dotenv
from workflow_agents.base_agents import RoutingAgent, KnowledgeAugmentedPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

texas_persona = "a Texas history expert"
texas_knowledge = "Rome, Texas is a small town in northeast Texas. It was named after Rome, Georgia."
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key, texas_persona, texas_knowledge)

europe_persona = "a European history expert"
europe_knowledge = "Rome, Italy is the capital of Italy and was the center of the Roman Empire. It has a rich history dating back thousands of years."
europe_agent = KnowledgeAugmentedPromptAgent(openai_api_key, europe_persona, europe_knowledge)

math_persona = "a mathematics expert"
math_knowledge = "To calculate total time for multiple stories, multiply the time per story by the number of stories."
math_agent = KnowledgeAugmentedPromptAgent(openai_api_key, math_persona, math_knowledge)

routing_agent = RoutingAgent(openai_api_key)

routing_agent.agents = [
    {
        "name": "Texas Expert",
        "description": "Expert on Texas geography and history, particularly towns and cities in Texas",
        "func": lambda x: texas_agent.respond(x)
    },
    {
        "name": "Europe Expert",
        "description": "Expert on European geography and history, particularly cities and countries in Europe",
        "func": lambda x: europe_agent.respond(x)
    },
    {
        "name": "Math Expert",
        "description": "Expert on mathematics, calculations, and numerical problems",
        "func": lambda x: math_agent.respond(x)
    }
]

test_prompts = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories"
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    response = routing_agent.route(prompt)
    print(f"Response: {response}")
    print("-" * 80)
