# Main workflow orchestration script for Email Router product development
# Coordinates multiple specialized agents to generate a comprehensive project plan

import os
from dotenv import load_dotenv
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

# Setup environment and API credentials
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Load product specification document
with open("Product-Spec-Email-Router.txt", "r") as f:
    product_spec = f.read()

# Instantiate agents
# Action Planning Agent: breaks down high-level requests into discrete steps
knowledge_action_planning = """You extract actionable steps from a user's request for technical project management. 
For a request to create a full product development plan, you should extract steps like:
1. Define user personas and user stories
2. Define product features based on user stories
3. Create detailed engineering tasks from features
List each step on a new line, numbered."""

action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager Agent: defines user personas and user stories
persona_product_manager = "a Product Manager responsible for defining user personas and creating user stories"

knowledge_product_manager = """As a Product Manager, you define user personas and create user stories based on product specifications. 
User stories should follow this structure: As a [type of user], I want [an action or feature] so that [benefit/value].
Focus on understanding user needs and translating them into clear, actionable stories.

Product Specification:
""" + product_spec

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager Evaluation Agent: validates user stories against required format
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    "You are an evaluation agent that checks the answers of other worker agents",
    "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value].",
    product_manager_knowledge_agent
)

# Program Manager Agent: defines product features from user stories
persona_program_manager = "a Program Manager responsible for defining product features"

knowledge_program_manager = """As a Program Manager, you define product features based on user stories and product requirements.
Features should be high-level capabilities that deliver value to users.
Each feature should include: Feature Name, Description, Key Functionality, and User Benefit."""

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_program_manager, knowledge_program_manager)

persona_program_manager_eval = "You are an evaluation agent that checks program manager outputs"

# Program Manager Evaluation Agent: validates feature format
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

# Development Engineer Agent: creates detailed engineering tasks
persona_dev_engineer = "a Development Engineer responsible for creating detailed technical tasks"

knowledge_dev_engineer = """As a Development Engineer, you create detailed engineering tasks from features and user stories.
Tasks must follow this EXACT format with ALL labeled fields:

Task ID: [unique identifier like T001, T002, etc.]
Task Title: [brief task name]
Related User Story: [reference to the user story this relates to]
Description: [detailed technical work explanation]
Acceptance Criteria: [specific completion requirements]
Estimated Effort: [time or complexity estimate]
Dependencies: [prerequisite tasks or 'None']

Each task MUST include all seven labeled fields above."""

dev_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

persona_dev_engineer_eval = "You are an evaluation agent that checks development engineer outputs"

# Development Engineer Evaluation Agent: validates task structure and completeness
dev_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    "The answer should be tasks following this exact structure with labeled fields: " \
    "Task ID: [unique identifier]\n" \
    "Task Title: [brief task name]\n" \
    "Related User Story: [reference to user story]\n" \
    "Description: [detailed technical work explanation]\n" \
    "Acceptance Criteria: [specific completion requirements]\n" \
    "Estimated Effort: [time or complexity estimate]\n" \
    "Dependencies: [prerequisite tasks or 'None']\n\n" \
    "Each task must have ALL these labeled fields.",
    dev_engineer_knowledge_agent
)

# Routing Agent: directs steps to the appropriate specialized agent
routing_agent = RoutingAgent(openai_api_key)

# Support functions: wrap agent execution with evaluation
def product_manager_support_function(query):
    # Execute product manager agent and validate output
    response = product_manager_knowledge_agent.respond(query)
    result = product_manager_evaluation_agent.evaluate(query)
    return result['final_response']

def program_manager_support_function(query):
    # Execute program manager agent and validate output
    response = program_manager_knowledge_agent.respond(query)
    result = program_manager_evaluation_agent.evaluate(query)
    return result['final_response']

def development_engineer_support_function(query):
    # Execute development engineer agent and validate output
    response = dev_engineer_knowledge_agent.respond(query)
    result = dev_engineer_evaluation_agent.evaluate(query)
    return result['final_response']

# Register specialized agents with routing agent
# Each agent has a description used for semantic matching with workflow steps
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

# Main workflow prompt defining the overall goal
workflow_prompt = """Create a comprehensive product development plan for the Email Router product. 
This should include: user stories for different user types, product features that fulfill those stories, 
and detailed engineering tasks to implement those features."""

print("=" * 100)
print("AGENTIC WORKFLOW FOR EMAIL ROUTER PRODUCT DEVELOPMENT")
print("=" * 100)
print()

# Generate workflow steps from high-level prompt
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

print("Workflow Steps Identified:")
for i, step in enumerate(workflow_steps, 1):
    print(f"  {i}. {step}")
print()
print("=" * 100)
print()

# Execute each workflow step
completed_steps = []
step_descriptions = []

for i, step in enumerate(workflow_steps, 1):
    print(f"EXECUTING STEP {i}: {step}")
    print("-" * 100)
    
    # Route step to appropriate agent based on semantic similarity
    result = routing_agent.route(step)
    completed_steps.append(result)
    step_descriptions.append(step)
    
    print(f"\nResult:\n{result}")
    print()
    print("=" * 100)
    print()

# Consolidate final deliverable with all components
print("WORKFLOW COMPLETE - FINAL OUTPUT:")
print("=" * 100)
print()
print("COMPREHENSIVE PROJECT PLAN FOR EMAIL ROUTER")
print()

# Extract and organize results by type
user_stories = []
product_features = []
engineering_tasks = []

for i, (step_desc, result) in enumerate(zip(step_descriptions, completed_steps)):
    if result:
        step_lower = step_desc.lower()
        # Identify user story steps
        if "user stor" in step_lower or "persona" in step_lower:
            user_stories.append(result)
        # Identify feature steps
        elif "feature" in step_lower:
            product_features.append(result)
        # Identify task steps
        elif "task" in step_lower or "engineering" in step_lower or "technical" in step_lower:
            engineering_tasks.append(result)

# Print consolidated output
if user_stories:
    print("=" * 100)
    print("USER STORIES")
    print("=" * 100)
    for story in user_stories:
        print(story)
        print()

if product_features:
    print("=" * 100)
    print("PRODUCT FEATURES")
    print("=" * 100)
    for feature in product_features:
        print(feature)
        print()

if engineering_tasks:
    print("=" * 100)
    print("ENGINEERING TASKS")
    print("=" * 100)
    for task in engineering_tasks:
        print(task)
        print()

print("=" * 100)
