# Install necessary packages (assuming this is run in a Colab environment)
# !pip install pdfminer.six pytesseract torch langchain langchain_openai langchain_community faiss-cpu python-docx

# Import necessary libraries
import os
import glob
from pdfminer.high_level import extract_text  # Import PDF text extraction function
from openpyxl import load_workbook  # Import Excel file handling
from PIL import Image  # Import PIL for image handling
import pytesseract  # Import pytesseract for OCR
from transformers import AutoTokenizer, AutoModel  # Import Hugging Face Transformers
import torch  # Import PyTorch
import numpy as np  # Import numpy for array operations
from google.colab import drive  # Import drive for Google Colab mounting
from langchain.text_splitter import CharacterTextSplitter  # Import text splitter
from langchain_openai import OpenAIEmbeddings  # Import OpenAI embeddings
import openai  # Import OpenAI library
from langchain.chains.question_answering import load_qa_chain  # Import QA chain loader
from langchain_openai import OpenAI  # Import OpenAI library (again?)
from langchain_community.vectorstores import FAISS  # Import FAISS for vector storage
from docx import Document  # Import Document handling for DOCX files

# Mount Google Drive for accessing files
drive.mount('/content/drive')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        text = extract_text(pdf_path)  # Use PDFMiner to extract text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

# Function to extract text from an Excel file
def extract_text_from_excel(excel_path):
    text = ""
    try:
        wb = load_workbook(excel_path)  # Load the Excel workbook
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            for row in ws.iter_rows(values_only=True):  # Iterate through rows
                for cell in row:
                    if isinstance(cell, str):
                        text += cell + " "  # Append cell content to text
    except Exception as e:
        print(f"Error extracting text from {excel_path}: {e}")
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(file_path):
    doc = Document(file_path)  # Load the DOCX document
    full_text = []
    for paragraph in doc.paragraphs:  # Iterate through paragraphs
        full_text.append(paragraph.text)  # Append paragraph text to list
    return '\n'.join(full_text)  # Join paragraphs with newline and return as single text

# Folder path where documents are stored
folder_path = "content/DRIVE_OF_YOUR_PATH"

# List all files in the folder
file_paths = glob.glob(os.path.join(folder_path, "*"))

raw_texts = []  # Initialize empty list to store raw text from documents

# Process each file and extract text
for file_path in file_paths:
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)  # Extract text from PDF
    elif file_path.endswith(".xlsx"):
        text = extract_text_from_excel(file_path)  # Extract text from Excel
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)  # Extract text from DOCX
    else:
        continue  # Skip files that are not PDF, XLSX, or DOCX
    raw_texts.append(text)  # Append extracted text to raw_texts list

# Initialize text splitter to chunk text into manageable pieces
text_splitter = CharacterTextSplitter(
    separator="\n",  # Separator between chunks
    chunk_size=4000,  # Size of each chunk
    chunk_overlap=300,  # Overlap between chunks
    length_function=len,  # Function to calculate length of text
)

split_texts = []  # Initialize list to store split texts

# Split each raw text into smaller chunks
for text in raw_texts:
    split_text = text_splitter.split_text(text)  # Split text using text splitter
    split_texts.extend(split_text)  # Extend the list with split texts

print(split_texts)  # Print the list of split texts

# Set OpenAI API key
openai.api_key = "OPENAI_API_KEY"
os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"

# Initialize OpenAIEmbeddings for generating embeddings
embeddings = OpenAIEmbeddings()

# Create FAISS index for efficient document similarity search
document_search = FAISS.from_texts(split_texts, embeddings)

# Load question answering chain using OpenAI's GPT-3.5 model
chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1, openai_api_key=openai.api_key), chain_type="stuff")

# System prompt for the question answering system
system_prompt = "You are a Q&A estate agent designed to answer customer questions accurately. You have access to information from provided XLSX files, including prices, availability, and contact details like phone numbers."

# Loop to continuously prompt for user queries
while True:
    query = input("")  # Prompt user to enter a query

    # Perform similarity search to find relevant documents
    docs = document_search.similarity_search(query)

    # Generate a response to the query using the question answering chain
    result = chain.invoke({'input_documents': docs, 'question': query})
    print(result['output_text'])  # Print the output text from the question answering system
