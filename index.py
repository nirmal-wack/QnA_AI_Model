import pymupdf as fitz  
from docx import Document
import pandas as pd
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import numpy as np
import gradio as gr
import glob
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import openai
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from urllib.parse import quote
from urllib.parse import unquote
import base64




# Data extraction functions

def extract_from_pdf(pdf_path):
    """Extract text and images from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    images = []
    for page in doc:
        text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return text, images

def extract_from_docx(docx_path):
    """Extract text and images from a DOCX file."""
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    images = []
    for rel in doc.part.rels:
        if "image" in rel:
            img = doc.part.rels[rel].target_part.blob
            images.append(Image.open(io.BytesIO(img)))
    return text, images

def extract_from_xlsx(xlsx_path):
    """Extract text from an XLSX file."""
    data = pd.read_excel(xlsx_path)
    return data.to_string(), []


folder_path = "FILE_PATH"
file_paths = glob.glob(os.path.join(folder_path, "*"))

# Extract raw texts from files
raw_texts = []
for file_path in file_paths:
    # print(file_path)
    if file_path.endswith(".pdf"):
        text, images = extract_from_pdf(file_path)
        # add_to_index('pdf', file_path, text_pdf, images_pdf)
    elif file_path.endswith(".xlsx"):
        text, _ = extract_from_xlsx(file_path)
        # add_to_index('xlsx', file_path, text_xlsx, [])
    elif file_path.endswith(".docx"):
        text, images = extract_from_docx(file_path)
        # add_to_index('docx', file_path, text_docx, images_docx)
    else:
        continue
    raw_texts.append((text, images))
    

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=4000,
    chunk_overlap=300,
    length_function=len,
)

split_texts = []
for text, images in raw_texts:
    split_text = text_splitter.split_text(text)
    split_texts.extend([(chunk, images) for chunk in split_text])

openai.api_key = "OPENAI_API_KEY"
os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts([text for text, _ in split_texts], embeddings)
chain = load_qa_chain(ChatOpenAI(model_name="gpt-4-turbo",temperature=0.1,
openai_api_key=openai.api_key), chain_type="stuff")
system_prompt = "give the response like QnA chatbot agent and dont show query as in answer."

def search_query(query):
    query = f"{system_prompt} {query}"
    docs = document_search.similarity_search(query)
    result = chain.invoke({'input_documents': docs, 'question': query})


    if any(keyword in query.lower() for keyword in ["plan","image", "picture", "photo", "diagram"]):
        folder_path = "Folder_path" f
        file_paths = glob.glob(os.path.join(folder_path, "*"))

        query_words = query.lower().split()
        image_files = [file for file in file_paths if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        matching_files = []

        # List all files in the specified folder
        for filename in image_files:
            # Check if any of the query words are in the filename
            if any(word in filename.lower() for word in query_words):
                matching_files.append(filename)


        if matching_files:
            # Sort by relevance (e.g., number of matching words, specific phrases, etc.)
            matching_files.sort(key=lambda x: sum(word in os.path.basename(x).lower() for word in query_words), reverse=True)
            most_relevant_image = matching_files[0]
            print(most_relevant_image)
            images = [{"url": most_relevant_image}]

            return {"output_text": "" if result else "No relevant content found.", "images": images}
        else:
            return {"output_text": "" if result else "No relevant content found.", "images": []}
        
        # print(matching_files)


    return result if result else {"output_text": "No relevant content found.", "images": []}



def display_results(query):
    # Simulate a search result based on the query
    result = search_query(query)
    text_response = result.get('output_text', 'No relevant content found.')
    images = result.get('images', [])

    return text_response, images if images else []

with gr.Blocks() as demo:

    gr.Markdown("### Proplens' AI Assistant")
    chatbot = gr.Chatbot(label="Chat History", elem_id="chatbot-output")
    msg = gr.Textbox(label="Your Message", placeholder="Type your question here...")
    clear = gr.Button("Clear Chat")

    def respond(message, chat_history):
        text_response, images = display_results(message)
        image_urls = [quote(image['url']) for image in images]
        chat_history.append((message, text_response))       
        
        
        if image_urls:
            for image_url in image_urls:
                # Each image URL is added as a separate message
                encoded_path = image_url
                decoded_path = unquote(encoded_path)
                with open(decoded_path, "rb") as image_file:
                    # Encode the binary data to base64
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

                chat_history.append((None, gr.HTML(f'<img src="data:image/png;base64,{encoded_string}">')))
                
        
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: (None, []), None, [msg, chatbot])

demo.launch()
