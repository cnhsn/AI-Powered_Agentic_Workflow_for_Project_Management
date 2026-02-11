# Test script for ActionPlanningAgent
# Demonstrates breaking down high-level requests into actionable steps

import os
from dotenv import load_dotenv
from workflow_agents.base_agents import ActionPlanningAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Define knowledge for step extraction
knowledge = "Extract and list the steps required to complete the task described in the user's prompt. Each step should be a clear, actionable item."

action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge)

# Test with a simple task to extract steps
prompt = "One morning I wanted to have scrambled eggs"
steps = action_planning_agent.extract_steps_from_prompt(prompt)

print(f"Prompt: {prompt}")
print("\nExtracted Action Steps:")
for i, step in enumerate(steps, 1):
    print(f"{i}. {step}")
