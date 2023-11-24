import streamlit as st
from transformers import pipeline
import PyPDF2
import re
import gdown
import os

# Function to download PDF file from Google Drive link
def download_pdf_from_drive(drive_link, save_path):
    gdown.download(drive_link, save_path, quiet=False)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Function to answer user's question
def answer_question(question, context):
    qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad', tokenizer='bert-large-uncased-whole-word-masking-finetuned-squad')
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def main():
    st.title("IIT Jodhpur Chatbot")

    # Google Drive link to the PDF file
    drive_link = 'https://drive.google.com/uc?id=12IuKxuRvwBJ5JPsLMbCrNGv5V_jINrmN'
    
    # Download PDF file from Google Drive
    pdf_path = 'temp.pdf'
    download_pdf_from_drive(drive_link, pdf_path)

    pdf_text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(pdf_text)

    user_question = st.text_input("Ask a question:")

    if user_question.lower() == 'exit':
        st.stop()

    # Answer user's question using PDF question-answering
    chatbot_response = answer_question(user_question, preprocessed_text)
    st.write("Chatbot:", chatbot_response)

    # Clean up: Remove the downloaded PDF file
    os.remove(pdf_path)

if __name__ == "__main__":
    main()
