import os
from dotenv import load_dotenv
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

with open("Product-Spec-Email-Router.txt", "r") as f:
    product_spec = f.read()

knowledge_action_planning = """You extract actionable steps from a user's request for technical project management. 
For a request to create a full product development plan, you should extract steps like:
1. Define user personas and user stories
2. Define product features based on user stories
3. Create detailed engineering tasks from features
List each step on a new line, numbered."""

action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

persona_product_manager = "a Product Manager responsible for defining user personas and creating user stories"

knowledge_product_manager = """As a Product Manager, you define user personas and create user stories based on product specifications. 
User stories should follow this structure: As a [type of user], I want [an action or feature] so that [benefit/value].
Focus on understanding user needs and translating them into clear, actionable stories.

Product Specification:
""" + product_spec

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    "You are an evaluation agent that checks the answers of other worker agents",
    "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value].",
    product_manager_knowledge_agent
)

persona_program_manager = "a Program Manager responsible for defining product features"

knowledge_program_manager = """As a Program Manager, you define product features based on user stories and product requirements.
Features should be high-level capabilities that deliver value to users.
Each feature should include: Feature Name, Description, Key Functionality, and User Benefit."""

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_program_manager, knowledge_program_manager)

persona_program_manager_eval = "You are an evaluation agent that checks program manager outputs"

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    "The answer should be product features that follow the following structure: " \
    "Feature Name: A clear, concise title that identifies the capability\n" \
    "Description: A brief explanation of what the feature does and its purpose\n" \
    "Key Functionality: The specific capabilities or actions the feature provides\n" \
    "User Benefit: How this feature creates value for the user",
    program_manager_knowledge_agent
)

persona_dev_engineer = "a Development Engineer responsible for creating detailed technical tasks"

knowledge_dev_engineer = """As a Development Engineer, you create detailed engineering tasks from features and user stories.
Tasks should be specific, actionable, and technical in nature.
Include implementation details, acceptance criteria, effort estimates, and dependencies."""

dev_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

persona_dev_engineer_eval = "You are an evaluation agent that checks development engineer outputs"

dev_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    "The answer should be tasks following this exact structure: " \
    "Task ID: A unique identifier for tracking purposes\n" \
    "Task Title: Brief description of the specific development work\n" \
    "Related User Story: Reference to the parent user story\n" \
    "Description: Detailed explanation of the technical work required\n" \
    "Acceptance Criteria: Specific requirements that must be met for completion\n" \
    "Estimated Effort: Time or complexity estimation\n" \
    "Dependencies: Any tasks that must be completed first",
    dev_engineer_knowledge_agent
)

routing_agent = RoutingAgent(openai_api_key)

def product_manager_support_function(query):
    response = product_manager_knowledge_agent.respond(query)
    result = product_manager_evaluation_agent.evaluate(query)
    return result['final_response']

def program_manager_support_function(query):
    response = program_manager_knowledge_agent.respond(query)
    result = program_manager_evaluation_agent.evaluate(query)
    return result['final_response']

def development_engineer_support_function(query):
    response = dev_engineer_knowledge_agent.respond(query)
    result = dev_engineer_evaluation_agent.evaluate(query)
    return result['final_response']

routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories",
        "func": lambda x: product_manager_support_function(x)
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining product features and capabilities. Does not create user stories or engineering tasks",
        "func": lambda x: program_manager_support_function(x)
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for creating detailed engineering tasks and technical implementation plans. Does not create user stories or features",
        "func": lambda x: development_engineer_support_function(x)
    }
]

workflow_prompt = """Create a comprehensive product development plan for the Email Router product. 
This should include: user stories for different user types, product features that fulfill those stories, 
and detailed engineering tasks to implement those features."""

print("=" * 100)
print("AGENTIC WORKFLOW FOR EMAIL ROUTER PRODUCT DEVELOPMENT")
print("=" * 100)
print()

workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

print("Workflow Steps Identified:")
for i, step in enumerate(workflow_steps, 1):
    print(f"  {i}. {step}")
print()
print("=" * 100)
print()

completed_steps = []

for i, step in enumerate(workflow_steps, 1):
    print(f"EXECUTING STEP {i}: {step}")
    print("-" * 100)
    
    result = routing_agent.route(step)
    completed_steps.append(result)
    
    print(f"\nResult:\n{result}")
    print()
    print("=" * 100)
    print()

print("WORKFLOW COMPLETE - FINAL OUTPUT:")
print("=" * 100)
print()
print(completed_steps[-1] if completed_steps else "No results generated")
print()
print("=" * 100)
