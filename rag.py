import streamlit as st
import os
import chromadb
from langchain.vectorstores.chroma import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
#from langchain.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import requests
import base64
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from PIL import Image
import shutil
#!pip install pypdf, langchain, chromadb, sentence-transformers
#!pip install --upgrade --quiet  langchain-google-genai pillow
from google.cloud import storage
import google.generativeai as genai

# from pydub.playback import play
# from pydub import AudioSegment
# from google.cloud import texttospeech
# from gtts import gTTS
temp = .4
GOOGLE_API_KEY = ""

# def text_to_speech (text):
#     # Language in which you want to convert
#     language = 'en'
#     myobj = gTTS(text=text, lang=language, slow=False)
#     myobj.save("welcome.mp3")
#     # Playing the converted file    
#     os.system("mpg321 welcome.mp3")



#For image
genai.configure(api_key=GOOGLE_API_KEY)
def gemini_resp (prompt, image_path): 
        response = model.generate_content(
            [Image.open(image_path),prompt],
            generation_config={
                "max_output_tokens": 2048,
                "temperature": temp,
                "top_p": 1,
                "top_k": 32
            },
        )
        return response.text

st.set_page_config(page_title="MultiLingual Health BOT", page_icon="rag_health.png")

CLOUD_STORAGE_BUCKET = 'rag_app_embedded_docs_hack2024'
chroma_dir = './chroma_db'

def download_blob(file_name, existing = False):
    gcs = storage.Client()
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(file_name)
    if existing:
        os.remove(file_name )
        shutil.rmtree(chroma_dir, ignore_errors=True)
    blob.download_to_filename(file_name)
    
    shutil.unpack_archive(file_name, chroma_dir)
    #blob.delete()

if os.path.exists(chroma_dir):
    pass
else:
    try:
        download_blob("chroma_db.zip", existing=False)
        #st.write("Chroma DB Repository downloaded successfully.")
    except Exception as e:
        st.write(f"Error in downloading Chroma DB Repository: {e}")


filter_list = []
lang_list = ['English', 'Japanese', 'Korean', 'Arabic', 'Bahasa Indonesia', 'Bengali', 'Bulgarian', 'Chinese', 'Croatian', 'Czech', \
'Danish', 'Dutch', 'Estonian',  'Finnish', 'French', 'German', 'Gujarati', 'Greek', 'Hebrew', 'Hindi', 'Hungarian', 'Italian', \
'Kannada', 'Latvian', 'Lithuanian', 'Malayalam', 'Marathi', 'Norwegian', 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Serbian', 'Slovak', \
'Slovenian', 'Spanish', 'Swahili', 'Swedish'
, 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian',  'Vietnamese']

#Make a retrieval object
client = chromadb.PersistentClient(path=chroma_dir)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(client=client, embedding_function=embedding_function, collection_name="articles_embeddings")
doc_list = list(set( [ meta['source'] for meta  in vectordb.get()['metadatas'] ]))
#st.write(f"Number Documents available for answering questions: { len(doc_list)}")


for doc in doc_list:
    filter_list.append({"source": doc})



# Sidebar contents
with st.sidebar:
    col1, col2  = st.columns([.25,.75])
    with col1:
        image = Image.open('rag_health.png')
        st.image(image)
    with col2:
        st.markdown("<h3 style='text-align: left'>MultiLingual Health BOT : Is your food healthy? and Health QA.  </h3>", unsafe_allow_html= True)
        st.write("This application uses Gemini Model. It can tell if your food is healthy by a picture of it. Also, it can get the answers from a repository of documents.")
    col1, col2, col3  = st.columns([.42,.29, .29])
    with col1:
        appmode = st.radio("**App Mode**", [ 'Is Your Food Healthy?', 'All Docs QA', 'Single Doc QA', 'General Health QA'], index=0, help="You may get answer from all documents or single document using Apllication Mode.")
        refresh_rep = st.button("Refresh Documents", help="Click only if new reposotory is available.")
    with col2:
        lang= st.selectbox("**Answer Language**", lang_list, index=0, help="Select a language to get the answer.")
        model_name = st.selectbox("**LLM Name**", ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro-latest" ], index=0, help="Select a model to get the answer.")
        
    with col3:
        num_chunks = st.number_input('**Number of Chunks**', min_value=1, max_value=15, value=3,step=1, help="Number of chunks to be used to answer.")
        temp= st.number_input('**Temperature**', min_value=0.0, max_value=1.0, value=0.4, step=0.1, help="Temperature for Gemini Model. Increase it for more creative answers.")  
      
    if refresh_rep:
            try:
                download_blob("chroma_db.zip")
                st.write("Chroma DB Repository downloaded successfully.")
            except Exception as e:
                st.write(f"Error in downloading Chroma DB Repository: {e}")
  
    st.write(f"You can upload an image of food item to get health advice. You can ask questions about any health related topic and app will \
    provide you answer from document repository along with references. You may select all documents or single document to get the answer. \
    You may select number of chunks to be used to answer the question. \
    Alternately you can use General Health QA to ask any health related question and health-bot will answer question without stored documents. \
    This bot is multi-lingual and can provide answer in multiple languages.")


    doc_names = None
    if appmode == 'All Docs QA':
        default_list = None
        st.markdown(f"**Documnets available for answering questions:**")
        doc_names_text= ""
        for i, doc in enumerate(doc_list):
            doc_names_text += f"{i+1}. {doc} \n"
        st.write(doc_names_text)
        #st.text_area(doc_names_text, height=200, help="List of documents available for answering questions.")
    elif appmode == 'Single Doc QA':
        doc_names = st.selectbox("Select Document", doc_list, help="Select a document to get the answer.")
        st.write(doc_names)
    elif appmode == 'General Health QA':
        st.write("You may ask any health related question. It will provide general health advice without stored documents repository.")

    else:
        default_list = None
        st.markdown(f"**Documnets available for answering questions:**")
        doc_names_text= ""
        for i, doc in enumerate(doc_list):
            doc_names_text += f"{i+1}. {doc} \n"
        st.write(doc_names_text)

    ## LLM model
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY, temperature=temp)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("For questions or feedback, please reach out to us. Email: hackathon2024gcp@gmail.com")

    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Copyright: © 2024 MultiLingual Health BOT. All rights reserved.")

    
    

if appmode == 'All Docs QA':
    retriever = vectordb.as_retriever(search_kwargs={"k": num_chunks })
elif appmode == 'Single Doc QA':
    retriever = vectordb.as_retriever(search_kwargs={"k": num_chunks , "filter": {'source': doc_names}})
    st.write(f"Selected document to get the answer: {doc_names}. Only this document will be used to answer the question.")
else:
    retriever = vectordb.as_retriever(search_kwargs={"k": num_chunks })

#Define the prompt template and retrieval chain
template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
answer:
"""
prompt = PromptTemplate.from_template(template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
if appmode == 'Is Your Food Healthy?':
    model = genai.GenerativeModel("gemini-pro-vision")
    uploaded_file = st.file_uploader("**Upload an image of your food item.**", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        user_prompt_content = "Study the image enclosed carefully. This is about a food item. Provide name and description of the food.\
        If you do nor see human food item. Say 'Not a food item'."
        image = Image.open(uploaded_file)
        st.image(image)
        food_desc = gemini_resp( user_prompt_content, uploaded_file)
        if food_desc == "Not a food item":
            st.write(f"This is not a food item as per Gemini Model. Please upload a food image.")
        else:
            if lang != 'English':
                try:
                    food_desc = llm.invoke(f"Translate the following text to {lang} :\n {food_desc}").content
                except Exception as e:
                    st.write(f"Could not be translated to {lang}. See English version  below.")
            st.write(f"**Food Description:**\n {food_desc}")
            health_tip=llm.invoke(f"Is {food_desc} healthy? Provide helath and nutrition related information about: {food_desc}").content
            if lang != 'English':
                try:
                    health_tip = llm.invoke(f"Translate the following text to {lang} :\n {health_tip}").content
                except Exception as e:
                    st.write(f"Could not be translated to {lang}. See English version  below.")
            st.write(f"**Health Advice:**\n {health_tip}")
elif appmode == 'General Health QA':
    general_question = st.text_input("**Type your general health question below and press enter.**",  value="", help="Type your general health question here. Example: I have cough and cold , what should I do?" ) 
    web_ref = st.checkbox("Show Web References", value=False, help="Show web references used to generate the answer.")
    if general_question == "":
        pass
        #st.write("Please type a general health question.")
    else:
        try:
            if web_ref:
                general_question = f"{general_question} \n Show web references used to generate the answer."
            llm_response = llm.invoke(f"As a general health practitioner, answer the given question. \
            If you don't have the answer or you don't understand the question, you can say that. Question:  { general_question}")
            #st.write(llm_response)
            ans = llm_response.content
           
          
            if lang == 'English':
                st.markdown(f"**QUESTION:** ")
                st.write(general_question)
                st.markdown(f"**ANSWER:**")
                st.write(ans)
                #text_to_speech(ans)
            else:
                try:
                    translated_question = llm.invoke(f"Translate the following text to {lang} :\n {general_question}").content
                    translated_ans = llm.invoke(f"Translate the following text to {lang} :\n {ans}").content
                    st.markdown(f"**QUESTION:** ")
                    st.write(translated_question)
                    st.markdown(f"**ANSWER:**")
                    st.write(translated_ans)
        
                except Exception as e:
                    st.markdown(f"**QUESTION:** ")
                    st.write(f"Answer could not be translated to {lang}. See English answer below.")  
                    st.write(general_question)
                    st.markdown(f"**ANSWER:**")
                    st.write(ans)
        except Exception as e:
            st.write(f"Error in using LLM: {e}")
   
else:

    question = st.text_input("**Type your question below.**",  value="", help="Type your question here. Example: How to prevent childhood obesity?" )
    col1, col2, col3, col4 = st.columns([.25,.30, .30, .15])
    with col1:
        show_sources = st.checkbox("Show Sources", value=True, help="Show source document and page numbers used to generate the answer.")
    with col2:
        print_text= st.checkbox("Show Text Chunks ", value=False, help="Print text chunks, used to answer the question.") 
    with col3:
        pass
    with col4:
        submit_button = st.button(label='SUBMIT', help='Click to submit question and get answer.')



    if submit_button:
        try:
            st.markdown(f"**QUESTION:** {question}")
            llm_response = retrieval_chain.invoke({"input":question})
            ans = llm_response['answer']
            st.markdown(f"**ANSWER:**")
            if lang == 'English':
                st.write(ans)
            else:
                try:
                    translated_ans = llm.invoke(f"Translate the following text to {lang} :\n {ans}")
                    st.write(translated_ans.content)
                except Exception as e:
                    st.write(f"Answer could not be translated to {lang}. See English answer below.")
                    st.write(ans)
                
    
            if show_sources:
                st.markdown(f"**Sources - File name(s) and page number(s) used to generate answer:**")
                for i, source in enumerate(llm_response["context"]):
                    st.write(f"{i+1}. {source.metadata['source']}    Page {source.metadata['page']}" )
            if print_text:
                st.markdown(f"**Text chunks used to generate answer:**" )
                for i, source in enumerate(llm_response["context"]):
                    st.write(f"*From File {source.metadata['source']}    Page: {source.metadata['page'] : }*" )
                    st.write(source.page_content)
        except Exception as e:
            st.write(f"Error in using LLM: {e}")
     

