import os
import streamlit as st
from youtube_transcript_api import TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Streamlit Page Config
st.set_page_config(page_title="YouTube Q&A with RAG", layout="wide")
st.title("YouTube Video Q&A with RAG")
st.write("Ask questions about a YouTube video transcript!")

# API Key Input

st.sidebar.header("API Key Settings")

if "api_key" not in st.session_state:
    st.session_state.api_key = None

api_key_input = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

if st.sidebar.button("Save API Key"):
    if (api_key_input.startswith("sk-") or api_key_input.startswith("sk-proj-")) and len(api_key_input) > 40:
        st.session_state.api_key = api_key_input
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        st.sidebar.success("API key saved successfully!")
    else:
        st.sidebar.error("Invalid API key. Must start with 'sk-' or 'sk-proj-'.")


if not st.session_state.api_key:
    st.warning("Please enter and save your OpenAI API key first.")
    st.stop()



# YouTube URL Input

youtube_url = st.text_input("Enter YouTube Video URL:")

if st.button("Load Transcript"):
    if youtube_url:
        try:
            # Load transcript
            loader = YoutubeLoader.from_youtube_url(youtube_url)
            transcript = loader.load()[0].page_content

           
            # Translate Transcript to English
    
            st.info("Detecting language and translating transcript into English...")
            translator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            translation_prompt = f"""
            You are a translator. Translate the following transcript into English.
            Keep the meaning intact, but make it clear and natural.

            Transcript:
            {transcript}
            """

            translated_transcript = translator.invoke(translation_prompt).content

            
            # Split transcript into chunks
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([translated_transcript])

           
            # Embeddings + FAISS Vector Store
          
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(chunks, embeddings)

            # Retriever
            retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

            # LLM
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

            # Prompt
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=['context', 'question']
            )

            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            # Chain
            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | llm | parser

            st.session_state.main_chain = main_chain
            st.success("Transcript processed successfully! Ask your questions below:")

        except TranscriptsDisabled:
            st.error("This video has no transcripts available.")
        except Exception as e:
            st.error(f"Error: {e}")



# Chat UI

if "main_chain" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask something about the video..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.main_chain.invoke(question)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
