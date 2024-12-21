from transformers import pipeline

def create_chatbot():
    """Create a chatbot using a pre-trained language model."""
    chatbot = pipeline("text-generation", model="distilgpt2")
    return chatbot

def answer_query(chatbot, query):
    """Generate an answer to a clinician's query."""
    response = chatbot(query, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

if __name__ == "__main__":
    chatbot = create_chatbot()
    query = "What are the symptoms of diabetes?"
    answer = answer_query(chatbot, query)
    print(f"Query: {query}\nAnswer: {answer}")
