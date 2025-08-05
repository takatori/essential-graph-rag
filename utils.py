from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import ollama


neo4j_driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size, overlap, split_on_whitespace_only=True):
    chunks = []
    index = 0

    while index < len(text):
        if split_on_whitespace_only:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index += chunk_size

    return chunks

def embed(texts):
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)
    return embeddings

def chunk_text(text, chunk_size,overlap, split_on_whitespace_only=True):
    chunks = []
    index = 0

    while index < len(text):
        if split_on_whitespace_only:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index += chunk_size

    return chunks


def chat(messages, model="llama3.2", config={}):
    response = ollama.chat(
        model=model,
        messages=messages,
        **config
    )
    return response['message']['content'].strip()


def tool_choice(messages, model="llama3.2", tools=[], config={}):
    print(f"tool choice Messages: {messages}")
    response = ollama.chat(
        model=model,
        messages=messages,
        tools=tools,
        **config
    )
    print(f"tool choice Response: {response}")
    return response['message']['tool_calls']