import os
import json
import requests
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from pinecone import Pinecone, ServerlessSpec  # Import the correct Pinecone class

# Load environment variables
load_dotenv()

# Initialize the parser
parser = LlamaParse(
    result_type='markdown',
)

# File extractor for parsing PDF
file_extractor = {".pdf": parser}
output_docs = SimpleDirectoryReader(input_files=['./data/report.pdf'], file_extractor=file_extractor)
docs = output_docs.load_data()

# Convert parsed docs to markdown text
md_text = ""
for doc in docs:
    md_text += doc.text

# Save the markdown to a file
with open('output.md', 'w') as file_handle:
    file_handle.write(md_text)

print("Markdown file created successfully")

# Chunking function
chunk_size = 1000
chunk_overlap = 200

def fixed_size_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Create fixed-size chunks of the markdown text
chunks = fixed_size_chunks(md_text, chunk_size, chunk_overlap)
print(f"Number of chunks: {len(chunks)}")

# Embedding with Jina API
jina_api_key = os.getenv('JINA_API_KEY')
headers = {
    'Authorization': f'Bearer {jina_api_key}',
    'Content-Type': 'application/json'
}
url = 'https://api.jina.ai/v1/embeddings'

# Embed chunks using Jina API
embedded_chunks = []
for chunk in chunks:
    payload = {
        'input': chunk,
        'model': 'jina-embeddings-v3'
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        embedded_chunks.append(response.json()['data'][0]['embedding'])
    else:
        print(f"Error embedding chunk: {response.status_code}, Response: {response.text}")

print(f"Number of embedded chunks: {len(embedded_chunks)}")

# Save embedded chunks to a JSON file
output_file = 'embedded_chunks.json'
data_to_save = {
    'chunks': chunks,
    'embeddings': embedded_chunks
}

with open(output_file, 'w') as f:
    json.dump(data_to_save, f)

print(f"Embedded chunks saved to {output_file}")

# Initialize Pinecone using the new method
pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY')
)

index_name = "pesuio-naive-rag"

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # Ensure embeddings have this dimensionality
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',  # Define cloud and region as per your setup
            region='us-west-2'
        )
    )

# Retrieve the index
index = pc.Index(index_name)

# Prepare data for Pinecone upsert
vectors_to_upsert = [
    {
        'id': f'chunk_{i}',
        'values': embedding,
        'metadata': {'text': chunk}
    }
    for i, (chunk, embedding) in enumerate(zip(chunks, embedded_chunks))
]

# Upsert embeddings to Pinecone
index.upsert(vectors=vectors_to_upsert)

print(f"Uploaded {len(vectors_to_upsert)} vectors to Pinecone")
