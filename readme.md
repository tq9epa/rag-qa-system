# README: Running the RAG-based QA System

This README explains how to set up and run the Retrieval-Augmented Generation (RAG) QA system on any computer. The instructions cover installing dependencies, running the script, and troubleshooting.

## Prerequisites

1. **Python Version**: Ensure you have Python 3.8 or later installed on your system.
2. **Internet Connection**: Required initially to download datasets and models.
3. **Package Manager**: `pip` or `conda` to install dependencies.

---

## Step 1: Clone the Repository

1. Clone the repository or copy the Python script (`rag_qa_system.py`) to your local machine.
2. Navigate to the project directory:
   ```bash
   cd /path/to/your/project
   ```

---

## Step 2: Set Up a Virtual Environment (Recommended)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

---

## Step 3: Install Dependencies

Run the following command to install all required libraries:

```bash
pip install transformers datasets sentence-transformers faiss-cpu
```

> **Note**: If you are using a GPU, install `faiss-gpu` instead of `faiss-cpu`:
> ```bash
> pip install faiss-gpu
> ```

---

## Step 4: Run the Script

Execute the script using:

```bash
python rag_qa_system.py
```

The script will:
1. Load the TREC dataset.
2. Encode documents for retrieval.
3. Use FAISS to find the most relevant document for each question.
4. Use a pre-trained T5 model to generate answers.
5. Print the questions, retrieved contexts, and generated answers.

---

## Expected Output

Sample output:

```plaintext
Question: What is the capital of France?
Retrieved Context: The capital of France is Paris.
Answer: Paris

Question: Who is the current president of the United States?
Retrieved Context: The president of the United States in 2023 is Joe Biden.
Answer: Joe Biden

...
```

---

## Offline Usage

To enable offline usage:

1. Save the TREC dataset locally:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("trec", trust_remote_code=True)
   dataset.save_to_disk("trec_dataset")
   ```

2. Save the models locally:
   ```bash
   transformers-cli download google/flan-t5-small
   transformers-cli download sentence-transformers/all-MiniLM-L6-v2
   ```

3. Modify the script to load the dataset and models from local paths:
   ```python
   dataset = load_from_disk("trec_dataset")
   model = AutoModelForSeq2SeqLM.from_pretrained("path/to/local/flan-t5-small")
   tokenizer = AutoTokenizer.from_pretrained("path/to/local/flan-t5-small")
   embedding_model = SentenceTransformer("path/to/local/all-MiniLM-L6-v2")
   ```

---

## Troubleshooting

1. **FAISS Installation Issues**:
   - Ensure you are installing the correct version (`faiss-cpu` or `faiss-gpu`).
   - If using `conda`, install FAISS via:
     ```bash
     conda install -c conda-forge faiss-cpu
     ```

2. **Performance Issues**:
   - For large datasets, consider optimizing FAISS retrieval settings.
   - If using a CPU, reduce the number of documents or use a smaller model.

3. **Dataset Download Errors**:
   - Ensure `trust_remote_code=True` is passed when loading the dataset.
   - Check your internet connection during the initial setup.

---

## Additional Notes

- **Customization**: Add more documents to the `documents` list for broader context.
- **Extensibility**: Replace `flan-t5-small` with a larger model like `flan-t5-large` for better answers.

---


