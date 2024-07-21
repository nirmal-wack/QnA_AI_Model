# QnA_AI_Model
Objective : Document Processing and Question Answering System

Key Features:

Document Processing: Extracts text from PDFs, Excel files (.xlsx), and Word documents (.docx).
Text Chunking: Segments large text into smaller chunks for efficient processing.
Embedding Generation: Utilizes OpenAI's embeddings to convert text chunks into numerical representations.
Question Answering: Implements a question answering pipeline using OpenAI's GPT-3.5 model, capable of retrieving relevant information from processed documents.
Interactive Interface: Provides a user-friendly interface for querying information from documents.
Technologies Used:

Python, PDFMiner, OpenPyXL, python-docx: for document handling and text extraction.
PyTesseract, PIL: for optical character recognition (OCR) from images within documents.
Transformers, Torch: for embedding generation and integration with language models.
OpenAI API: for advanced question answering capabilities using GPT-3.5.
FAISS: for efficient similarity search based on document embeddings.
Usage Example:

Upload documents to the specified folder.
Run the script to process and extract information from documents.
Enter queries to retrieve specific details from the processed documents.
Project Purpose:
This project serves as a robust framework for businesses or individuals needing to automate document processing tasks and enhance customer service through AI-driven question answering. By leveraging modern NLP techniques and embeddings, it provides accurate and efficient document analysis and retrieval of relevant information.

Future Enhancements:

Support for more document formats.
Integration with cloud storage services.
Enhanced error handling and scalability improvements.
