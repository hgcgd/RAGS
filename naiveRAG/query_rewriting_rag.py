import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
from groq import Groq
from typing import List, Set, Dict
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "pesuio-naive-rag"  # Replace with your Pinecone index name
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

def generate_query_variations(query: str) -> List[str]:
    """Generate multiple variations of the input query using Groq."""
    prompt = f"""Generate 5 different variations of the following query. 
    Make the variations diverse - some should be more specific, some more general, 
    some should rephrase key concepts. Return only the queries, one per line.
    
    Original query: {query}
    """
    
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-70b-versatile",
    )
    
    # Split the response into individual queries and remove any empty lines
    variations = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
    # Add the original query to the variations
    variations.append(query)
    return variations

def embed_query(query: str) -> List[float]:
    """Embed a single query using Jina API."""
    payload = {
        'input': query,
        'model': 'jina-embeddings-v2-base-en'
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"Error embedding query: {response.status_code}")
        return []
    return response.json()['data'][0]['embedding']

def retrieve_nearest_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """Retrieve nearest chunks for a single query."""
    query_embedding = embed_query(query)
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return query_response['matches']

def retrieve_with_query_variations(original_query: str, top_k: int = 5) -> List[Dict]:
    """Retrieve chunks using multiple query variations."""
    # Generate query variations
    query_variations = generate_query_variations(original_query)
    print("\nQuery variations:")
    for i, var in enumerate(query_variations, 1):
        print(f"{i}. {var}")
    
    # Retrieve chunks for each variation
    all_chunks = []
    seen_texts = set()
    
    for query in query_variations:
        chunks = retrieve_nearest_chunks(query, top_k)
        for chunk in chunks:
            # Use text content as a key to avoid duplicates
            chunk_text = chunk['metadata']['text']
            if chunk_text not in seen_texts:
                seen_texts.add(chunk_text)
                all_chunks.append(chunk)
    
    # Sort combined results by score
    all_chunks.sort(key=lambda x: x['score'], reverse=True)
    return all_chunks

def generate_response(query: str, context: str) -> str:
    """Generate a response using the combined context."""
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
    The context was retrieved using multiple variations of the user's query to ensure comprehensive coverage. 
    If the answer is not in the context, say "I don't have enough information to answer that question."
    
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
    
    # Retrieve chunks using query variations
    nearest_chunks = retrieve_with_query_variations(query)
    
    # Combine all unique chunks into context
    context = "\n".join([chunk['metadata']['text'] for chunk in nearest_chunks])
    
    # Generate final response
    rag_response = generate_response(query, context)
    
    # Print results
    print(f"\nOriginal Query: {query}")
    print("\nRAG Response:")
    print(rag_response)
    print("\nRetrieved unique chunks:")
    for i, chunk in enumerate(nearest_chunks, 1):
        print(f"\n{i}. Score: {chunk['score']:.4f}")
        print(f"Text: {chunk['metadata']['text'][:200]}...")  # Display first 200 characters