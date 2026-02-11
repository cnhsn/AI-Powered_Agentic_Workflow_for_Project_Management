from openai import OpenAI
import numpy as np


class DirectPromptAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def respond(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def respond(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}. Forget all previous context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge
        self.client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
    
    def respond(self, prompt):
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
        prompt_embedding = self.get_embedding(prompt)
        similarities = []
        for doc in self.knowledge_documents:
            doc_embedding = self.get_embedding(doc)
            similarity = np.dot(prompt_embedding, doc_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc, similarity))
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
        iteration_count = 0
        current_prompt = prompt
        
        for i in range(self.max_interactions):
            iteration_count += 1
            worker_response = self.agent_to_evaluate.respond(current_prompt)
            
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
            
            if "yes" in evaluation_result.lower() and "no" not in evaluation_result.lower()[:evaluation_result.lower().index("yes") if "yes" in evaluation_result.lower() else 0]:
                return {
                    "final_response": worker_response,
                    "evaluation": evaluation_result,
                    "iterations": iteration_count
                }
            
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
            
            current_prompt = f"{prompt}\n\nPrevious response: {worker_response}\n\nCorrection needed: {correction_instructions}\n\nPlease provide an improved response."
        
        return {
            "final_response": worker_response,
            "evaluation": evaluation_result,
            "iterations": iteration_count
        }


class RoutingAgent:
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
        prompt_embedding = self.get_embedding(prompt)
        best_similarity = -1
        best_agent = None
        
        for agent in self.agents:
            agent_description_embedding = self.get_embedding(agent["description"])
            similarity = np.dot(prompt_embedding, agent_description_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(agent_description_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_agent = agent
        
        if best_agent:
            return best_agent["func"](prompt)
        return None


class ActionPlanningAgent:
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
        steps = [line.strip() for line in response_text.split("\n") if line.strip() and not line.strip().startswith("#")]
        return steps
