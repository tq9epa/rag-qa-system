from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Load the TREC dataset
trec = load_dataset("trec", trust_remote_code=True)

# Step 2: Prepare documents for retrieval
documents = [
    "The capital of France is Paris.",
    "The highest mountain in the world is Mount Everest.",
    "The solar system has eight planets.",
    "The largest mammal on Earth is the blue whale.",
    "The president of the United States in 2023 is Joe Biden."
]

# Step 3: Encode the documents for retrieval
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(documents)

# Step 4: Index the documents using FAISS
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

# Function to retrieve the most relevant context
def retrieve_context(question, top_k=1):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(question_embedding, k=top_k)
    retrieved_contexts = [documents[idx] for idx in indices[0]]
    return " ".join(retrieved_contexts)

# Step 5: Initialize a T5 model for answer generation
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    input_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids
    outputs = t5_model.generate(input_ids)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 6: Interactive QA Example
questions = [
    "What is the capital of France?",
    "Who is the current president of the United States?",
    "What is the highest mountain in the world?",
    "How many planets are in the solar system?",
    "What is the largest mammal on Earth?"
]

for question in questions:
    print(f"Question: {question}")
    context = retrieve_context(question)
    print(f"Retrieved Context: {context}")
    answer = generate_answer(question, context)
    print(f"Answer: {answer}\n")
