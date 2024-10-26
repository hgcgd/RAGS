
import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "pesuio-rag"  # Replace with your Pinecone index name
index = pc.Index(index_name)
# Jina API setup
jina_api_key = os.getenv('JINA_API_KEY')
headers = {
    'Authorization': f'Bearer {jina_api_key}',
    'Content-Type': 'application/json'
}
url = 'https://api.jina.ai/v1/embeddings'
# Initialize Groq client
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
def retrieve_nearest_chunks(query, top_k=5):
    # Embed the query using Jina API
    payload = {
        'input': query,
        'model': 'jina-embeddings-v2-base-en'
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"Error embedding query: {response.status_code}")
        return []
    query_embedding = response.json()['data'][0]['embedding']
    # Query Pinecone for nearest vectors
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return query_response['matches']
def generate_response(query, context):
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. If the answer is not in the context, say "I don't have enough information to answer that question."
Context:
{context}
User's question: {query}
Answer:"""
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-70b-versatile",
    )
    return response.choices[0].message.content
if __name__ == "__main__":
    query = "What is the title of the report?"
    nearest_chunks = retrieve_nearest_chunks(query)
    context = "\n".join([chunk['metadata']['text'] for chunk in nearest_chunks])
    
    rag_response = generate_response(query, context)
    print(f"\nQuery: {query}")
    print("\nRAG Response:")
    print(rag_response)
    print("\nRetrieved chunks:")
    for i, chunk in enumerate(nearest_chunks, 1):
        print(f"\n{i}. Score: {chunk['score']:.4f}")
        print(f"Text: {chunk['metadata']['text'][:200]}...")  # Display first 200 characters