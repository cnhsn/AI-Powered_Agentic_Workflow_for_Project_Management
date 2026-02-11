# Base agent implementations for agentic workflow system
# Provides different types of agents with varying levels of knowledge augmentation and evaluation

from openai import OpenAI
import numpy as np


class DirectPromptAgent:
    """
    Basic agent that sends prompts directly to the LLM without modification.
    Uses only the model's built-in knowledge.
    """
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def respond(self, prompt):
        # Send prompt to LLM without any modifications
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class AugmentedPromptAgent:
    """
    Agent that uses a persona to shape response tone and style.
    Still relies on the LLM's general knowledge but with persona-specific framing.
    """
    def __init__(self, openai_api_key, persona):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def respond(self, prompt):
        # Apply persona as system message to shape response style
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}. Forget all previous context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class KnowledgeAugmentedPromptAgent:
    """
    Agent that uses specific knowledge and a persona to generate responses.
    Explicitly instructed to rely on provided knowledge rather than general LLM knowledge.
    """
    def __init__(self, openai_api_key, persona, knowledge):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def respond(self, prompt):
        # Inject persona and knowledge into system message to guide response
        system_message = f"You are {self.persona} knowledge-based assistant. Forget all previous context. Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}. Answer the prompt based on this knowledge, not your own."
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class RAGKnowledgePromptAgent:
    """
    Retrieval-Augmented Generation agent that retrieves relevant knowledge from documents
    using semantic similarity before generating responses.
    """
    def __init__(self, openai_api_key, persona, knowledge_documents):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge_documents = knowledge_documents
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def get_embedding(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    
    def retrieve_relevant_knowledge(self, prompt, top_k=2):
        # Compute cosine similarity between prompt and each document
        prompt_embedding = self.get_embedding(prompt)
        similarities = []
        for doc in self.knowledge_documents:
            doc_embedding = self.get_embedding(doc)
            # Cosine similarity = dot product / (norm1 * norm2)
            similarity = np.dot(prompt_embedding, doc_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc, similarity))
        # Sort by similarity descending and return top k documents
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in similarities[:top_k]]
    
    def respond(self, prompt):
        relevant_knowledge = self.retrieve_relevant_knowledge(prompt)
        knowledge_context = "\n".join(relevant_knowledge)
        system_message = f"You are {self.persona} knowledge-based assistant. Forget all previous context. Use only the following knowledge to answer, do not use your own knowledge: {knowledge_context}. Answer the prompt based on this knowledge, not your own."
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class EvaluationAgent:
    """
    Agent that iteratively evaluates and corrects another agent's responses.
    Uses a feedback loop to improve quality until criteria are met or max iterations reached.
    """
    def __init__(self, openai_api_key, persona, evaluation_criteria, agent_to_evaluate, max_interactions=5):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max_interactions
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def evaluate(self, prompt):
        # Iterative evaluation and correction loop
        iteration_count = 0
        current_prompt = prompt
        
        for i in range(self.max_interactions):
            iteration_count += 1
            # Get response from worker agent
            worker_response = self.agent_to_evaluate.respond(current_prompt)
            
            # Evaluate the response against criteria
            evaluation_prompt = f"Evaluate the following response based on these criteria: {self.evaluation_criteria}\n\nResponse: {worker_response}\n\nDoes this response meet the criteria? Answer with 'Yes' or 'No' and explain why."
            evaluation_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0
            )
            evaluation_result = evaluation_response.choices[0].message.content
            
            # Check if response passes evaluation
            if "yes" in evaluation_result.lower() and "no" not in evaluation_result.lower()[:evaluation_result.lower().index("yes") if "yes" in evaluation_result.lower() else 0]:
                return {
                    "final_response": worker_response,
                    "evaluation": evaluation_result,
                    "iterations": iteration_count
                }
            
            # Generate correction instructions for next iteration
            correction_prompt = f"The following response did not meet the criteria: {self.evaluation_criteria}\n\nResponse: {worker_response}\n\nEvaluation: {evaluation_result}\n\nProvide specific instructions on how to correct this response."
            correction_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0
            )
            correction_instructions = correction_response.choices[0].message.content
            
            # Update prompt with correction feedback for next iteration
            current_prompt = f"{prompt}\n\nPrevious response: {worker_response}\n\nCorrection needed: {correction_instructions}\n\nPlease provide an improved response."
        
        # Return last response if max iterations reached
        return {
            "final_response": worker_response,
            "evaluation": evaluation_result,
            "iterations": iteration_count
        }


class RoutingAgent:
    """
    Agent that routes prompts to the most appropriate specialized agent
    based on semantic similarity between the prompt and agent descriptions.
    """
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.agents = []
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def get_embedding(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    
    def route(self, prompt):
        # Find the best agent using cosine similarity
        prompt_embedding = self.get_embedding(prompt)
        best_similarity = -1
        best_agent = None
        
        for agent in self.agents:
            agent_description_embedding = self.get_embedding(agent["description"])
            # Compute cosine similarity between prompt and agent description
            similarity = np.dot(prompt_embedding, agent_description_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(agent_description_embedding)
            )
            
            # Track the agent with highest similarity score
            if similarity > best_similarity:
                best_similarity = similarity
                best_agent = agent
        
        # Execute the best matching agent's function
        if best_agent:
            return best_agent["func"](prompt)
        return None


class ActionPlanningAgent:
    """
    Agent that breaks down high-level requests into discrete, actionable steps.
    Parses and cleans the LLM's response to extract a structured list of steps.
    """
    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def extract_steps_from_prompt(self, prompt):
        system_message = f"You are an Action Planning Agent. {self.knowledge}"
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content
        # Parse response into clean steps, filtering out empty lines and headers
        steps = [line.strip() for line in response_text.split("\n") if line.strip() and not line.strip().startswith("#")]
        return steps
