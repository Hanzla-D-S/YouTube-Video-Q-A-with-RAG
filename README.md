# YouTube Q&A with RAG

This is a Streamlit application that lets you interact with YouTube video transcripts using Retrieval-Augmented Generation (RAG).  
You can enter a YouTube video URL, automatically fetch its transcript, translate it into English (if needed), and then ask questions about the video.

## Features
- Fetches YouTube transcripts automatically
- Translates transcripts into English
- Splits text into manageable chunks for better retrieval
- Embeds transcripts into a FAISS vector database
- Uses OpenAI models for answering questions
- Simple chat-style interface built with Streamlit

## Requirements
- Python 3.9+
- An OpenAI API key

## Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/Hanzla-D-S/YouTube-Video-Q-A-with-RAG)
   cd your-repo-name
## Create a virtual environment:
- python -m venv venv
- venv\Scripts\activate
## Install the dependencies:
- pip install -r requirements.txt
## Run the Streamlit app:
- streamlit run app.py


