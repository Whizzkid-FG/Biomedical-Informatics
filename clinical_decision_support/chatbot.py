from transformers import pipeline, Pipeline
import logging

logging.basicConfig(level=logging.INFO)

def create_chatbot() -> Pipeline:
    """Create a chatbot using a pre-trained language model."""
    chatbot = pipeline("text-generation", model="distilgpt2")
    return chatbot

def answer_query(chatbot: Pipeline, query: str) -> str:
    """Generate an answer to a clinician's query."""
    try:
        response = chatbot(query, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, I couldn't process your request."

if __name__ == "__main__":
    try:
        chatbot = create_chatbot()
        query = "What are the symptoms of diabetes?"
        answer = answer_query(chatbot, query)
        print(f"Query: {query}\nAnswer: {answer}")
    except Exception as e:
        logging.error(f"Error in chatbot execution: {e}")
