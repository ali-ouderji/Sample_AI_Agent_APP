# Sample_AI_Agent_APP

# Rental Coordinator AI Agent

This project is an AI-powered rental inquiry assistant for forklift hire businesses. It guides clients through the quoting process by asking some questions and matching the request to the most suitable forklift based on lifting specs, site conditions, and availability.

---

## What It Does

- Accepts natural language rental inquiries (e.g. “Chasing a 2.5T forklift”)
- Asks spec questions to find out users need:
   1) Maximum lifting weight
   2) Maximum load size
   3) Indoor, outdoor, or both
   4) Ground conditions
   5) Turning space or access limits
   6) Rental duration? 2 month
   7) Delivery location
- Matches requirements to the closest forklift(s) in the spec sheet
- Returns a rental quote including:
  - Recommended forklift(s) with specs
  - Rental rates (based on pricing sheet)
  - Delivery note

- This app has been deployed as a web app using **Streamlit Cloud**
- The link to the app: https://sample-ai-agent-app-rental-use-case.streamlit.app/

---

## Requirements 

- **Python 3.10+**
- **Streamlit** for the frontend
- **LangChain** with OpenAI & Pinecone for conversational flow and vector retrieval (optional)
- **Pandas/Numpy** for data processing
- **Unstructured / PDF parsing** for brochure intelligence
- **Tesseract + pdf2image** for OCR if needed

---

Install dependencies using:

```bash
pip install -r requirements.txt

## Details about the python codes and model architecture:

# create_vector_db_and_qa_chain.py
This script sets up a vector database and a question-answering (QA) chain for a forklift rental coordination AI agent.
Below are the key components and steps involved:

File Preprocessing
- File Consolidation: All .csv and .pdf files are merged into a single .pdf for simplified loading and parsing.
- PDF Loading: Since the PDFs contain a mix of text, tables, and images, the script uses UnstructuredPDFLoader from langchain_community.document_loaders to effectively parse all content types.
- Chunking: The RecursiveCharacterTextSplitter is used to break down the loaded documents into manageable text chunks for embedding and retrieval.

Embedding and Vector Store
- Embeddings are generated using the text-embedding-3-large model from OpenAI.
- Pinecone is used for storing and querying the embeddings in a cloud-deployed vector database.

Question-Answering Chain
LLM Configuration: The QA chain uses ChatOpenAI with the gpt-4 model and temperature=0 for high factual accuracy and minimal creativity.

Note: This script does not require an API key for running the model.

# rental_agen.py
Custom Tools for Forklift Rentals
Core Features:
Weight Parsing: A custom function parse_weight_to_tons() is implemented to extract the maximum lifting weight from LLM responses for validation.

Agent Tools:
- print_finish: Finalizes the forklift rental request with a structured summary string.
- get_max_lifting_weight: Extracts only the maximum lifting weight after rental request finalization.

Agent Architecture
The AI agent is constructed using RunnableWithMessageHistory, enabling conversational memory across user interactions.

Prompt Structure:
python
Copy
Edit
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """SYSTEM PROMPT HERE"""
    ),
    MessagesPlaceholder(variable_name="history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

- agent_scratchpad provides visibility into how the LLM uses tools and forms its chain of thought.


