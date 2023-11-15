import streamlit as st
import pickle
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
import os


# Initialize Streamlit sidebar
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])



# Sidebar contents
with st.sidebar:
    st.title('URL-Based LLM Chatbot 🤖')
    key = st.text_input("Add your API Key")
    print(key)
    url = st.text_input("Add your url of webpage here:")
    os.environ["OPENAI_API_KEY"] = key
    st.subheader("Your Conversation")
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.expander("User"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.expander("Assistant"):
                st.markdown(message["content"])
    st.markdown('''
    ## About APP:

    The app's primary resource is utilized to create::

    - [streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/yashwant-rai-2157aa28b)

    ''')

    st.write('💡All about pdf-based chatbot, created by Yashwant Rai')


# Define CSS styles for the "New Chat" button
button_style = (
    "position: absolute; top: 10px; left: 10px; "
    "z-index: 1000; padding: 10px; background-color: #4CAF50; "
    "color: white; border: none; cursor: pointer;"
)

# "New Chat" button
if st.button("New Chat", key="start_new_chat", help="Click to start a new chat"):
    st.session_state.is_chatting = True
    st.session_state.messages = []

# Define a variable to store the chat history in the Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to extract text content from a URL
def get_text_from_url(url):
    try:
        response = requests.get(url)
        text = response.text
        return text
    except Exception as e:
        st.error(f"Error fetching URL content: {e}")
        return None


# Main function
def main():
    if url:
        text = get_text_from_url(url)

        if text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # # embeddings
            store_name = "URL_Content"

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            query = st.text_input("Ask questions about the URL content:")
            # Check if the "New Chat" button was clicked to reset the chat state
            if st.session_state.start_new_chat:
                st.session_state.is_chatting = True
                st.session_state.messages = []  # Reset chat history


            # Memory initialization
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            if query:
                chat_history = []
                with st.chat_message("user"):
                    st.markdown(query)
                st.session_state.messages.append({"role": "user", "content": query})

                custom_template = """Your custom prompt template here"""

                CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

                llm = OpenAI(temperature=0)

                qa = ConversationalRetrievalChain.from_llm(
                    llm,
                    VectorStore.as_retriever(),
                    condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                    memory=memory
                )
                response = qa({"question": query, "chat_history": chat_history})

                with st.chat_message("assistant"):
                    st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                st.session_state.chat_history.append((query, response))


# In the sidebar, display the chat history
with st.sidebar:
    st.title('Your Chat_History')
    for i, (user_msg, bot_response) in enumerate(st.session_state.chat_history):
        with st.expander(f"Chat {i + 1}"):
            st.markdown(f"User: {user_msg}")
            st.markdown(f"Assistant: {bot_response}")
            
                

# Run the main function
if __name__ == '__main__':
    main()
