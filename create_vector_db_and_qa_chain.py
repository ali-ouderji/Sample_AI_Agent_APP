import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from tqdm import tqdm
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from tqdm import tqdm


# Load environment variables
load_dotenv()

# ========== Configuration ==========
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
BATCH_SIZE = 100
K_RETRIEVE = 3
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_INDEX_NAME = 'forklift-rental-db'
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ========== Document Reading ==========
def sanitize_metadata(metadata):
    """Keep only values Pinecone accepts: str, int, float, bool, or list of str"""
    allowed_types = (str, int, float, bool)
    return {
        k: v for k, v in metadata.items()
        if isinstance(v, allowed_types) or (isinstance(v, list) and all(isinstance(i, str) for i in v))
    }

def load_pdf_documents(filepath: str):
    try:
        loader = UnstructuredPDFLoader(
            file_path=filepath,
            mode="elements",
            strategy="hi_res",
            infer_table_structure=True,
            include_metadata=True
        )
        docs = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        docs = []

    processed_docs = []
    for doc in docs:
        # Replace page_content with table HTML if present
        table_html = doc.metadata.get("text_as_html")
        if table_html:
            doc.page_content = table_html

        # Sanitize metadata
        doc.metadata = sanitize_metadata(doc.metadata)

        processed_docs.append(doc)

    return processed_docs


def load_excel_document(filepath: str):
    """Loads an Excel document from the specified file."""
    # loader = CSVLoader(file_path=filepath)
    loader = UnstructuredExcelLoader(file_path=filepath)
    return loader.load()


def load_document(filepath: str):
    """Loads documents from the specified file."""
    if filepath.lower().strip().endswith('.xlsx'):
        return load_excel_document(filepath)
    elif filepath.lower().strip().endswith('.pdf'):
        return load_pdf_documents(filepath)
    else:
        raise ValueError(f"Invalid file format: {filepath}. Expected a .pdf or .xlsx file.")


# ========== Text Splitting ==========
def split_documents(processed_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in processed_docs:
        # Skip splitting of table documents to avoid breaking HTML
        if "text_as_html" in doc.metadata:
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))

    return chunks


# ========== Embedding + Vector Store ==========
def create_vector_store(docs):
    """Generates embeddings and creates a Pinecone vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",  # 3072 dims
                                  api_key=OPENAI_API_KEY)
    
    # Batch upload loop
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Uploading batches"):
        batch = docs[i:i + BATCH_SIZE]
        try:
            PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            )
            print(f"Uploaded batch {i // BATCH_SIZE + 1}")
        except Exception as e:
            print(f"Failed batch {i // BATCH_SIZE + 1}: {e}")

    # return PineconeVectorStore(embedding=embeddings, index_name=PINECONE_INDEX_NAME)

# ========== Create or Get Vector Store ==========
def create_vector_db(input_file=None):
    if input_file is None:
        input_file= r"C:\SampleTest\data\final_file\final_data_for_vector_srore.pdf"

    docs = load_document(input_file)  
    chunks = split_documents(processed_docs=docs)
    vector_db = create_vector_store(chunks)

    return vector_db  

def get_vector_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",  # 3072 dims
                                  api_key=OPENAI_API_KEY)
    vectore_db =  PineconeVectorStore(embedding=embeddings, index_name=PINECONE_INDEX_NAME)
    return vectore_db

# ========== QA Chain Setup ==========
def create_qa_chain(vectorstore):
    """Creates a QA chain using GPT-4 and the vector retriever WITHOUT MEMORY."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVE})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

prompt_template = '''
You are a helpful AI assistant to answer queries about renting forklifts from our company.
Answer my questions based on the ifnormation provided and the older conversation. Do not make up answers.
If you do not know the answer to a question, just say "I don't know".

{context}

Given the following conversation and a follow up question, answer the question.

{chat_history}

question: {question}
'''

PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "chat_history", "question"]
        )

LLM = ChatOpenAI(model_name="gpt-4", temperature=0)

MEMORY = ConversationBufferMemory(
                                memory_key="chat_history",
                                max_len=50, return_messages=True,
                                output_key='answer')


def create_qa_chain_with_memory(vectorstore, memory=MEMORY, llm=LLM, prompt=PROMPT):
    """Creates a Conversational QA chain WITH MEMORY using GPT-4."""

    retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVE})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        # get_chat_history=get_chat_history, # raise error
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt})
    
    return qa_chain


if __name__ == '__main__':

    # Create vector stores
    # vector_db = create_vector_db()
    vector_db = get_vector_db()

    # Create question-answering chain with memory
    qa_chain_with_memory = create_qa_chain_with_memory(vector_db)

    # First query
    query1 = 'A price quote for the PG 4t Forklift'
    answer1 = qa_chain_with_memory.invoke({'question':query1, 
                                          'chat_history': MEMORY.chat_memory.messages})
    print(answer1['answer'])

    # Second query: making sure the memory works well 
    query2 = 'What was my last question?'
    answer2 = qa_chain_with_memory.invoke({'question':query2, 
                                           'chat_history': MEMORY.chat_memory.messages})
    print(answer2['answer'])

    # Third query
    query3 = 'List the specification of the first forklif with load capacity above 4000 kg'
    answer3 = qa_chain_with_memory.invoke({'question':query3, 
                                           'chat_history': MEMORY.chat_memory.messages})
    print(answer3['answer'])

    
    


    

 
      







