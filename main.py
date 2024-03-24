import streamlit as st
from st_clickable_images import clickable_images
import pandas as pd
from pytube import YouTube
import os
import requests
from time import sleep 
import assemblyai as aai
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")


def get_pdf_text(pdf):
    text = ""
    
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def generate_gemini_content(transcript_text, prompt):
    safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    ]
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt+transcript_text, safety_settings=safety_settings)
    
    return response.text

def questionnaire_content(transcript_text, prompt_questionnaire, vectorstore):
    safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    ]
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt_questionnaire+transcript_text, safety_settings=safety_settings)
    
    return response.text

transcriber = aai.Transcriber()


config = aai.TranscriptionConfig(speaker_labels=True)

upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

headers = {
    "authorization": "13910991857543d796cff5baf3cf8ae0",
    "content-type": "application/json"
}

@st.cache_data
def save_audio(url):
    yt = YouTube(url)
    try:
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download()
    except:
        return None, None, None
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    os.rename(out_file, file_name)
    print(yt.title + " has been successfully downloaded.")
    print(file_name)
    return yt.title, file_name, yt.thumbnail_url

@st.cache_data
def upload_to_AssemblyAI(save_location):
    CHUNK_SIZE = 5242880
    print(save_location)

    def read_file(filename):
        with open(filename, 'rb') as _file:
            while True:
                print("chunk uploaded")
                data = _file.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    upload_response = requests.post(
        upload_endpoint,
        headers=headers, data=read_file(save_location)
    )
    print(upload_response.json())

    if "error" in upload_response.json():
        return None, upload_response.json()["error"]

    audio_url = upload_response.json()['upload_url']
    print('Uploaded to', audio_url)

    return audio_url, None

@st.cache_data
def start_analysis(audio_url):
    print(audio_url)

    ## Start transcription job of audio file
    data = {
        'audio_url': audio_url,
        'iab_categories': True,
        'content_safety': True,
        "summarization": True,
        "summary_model": "informative",
        "summary_type": "gist"
    }

    transcript_response = requests.post(transcript_endpoint, json=data, headers=headers)
    print(transcript_response.json())

    if 'error' in transcript_response.json():
        return None, transcript_response.json()['error']

    transcript_id = transcript_response.json()['id']
    polling_endpoint = transcript_endpoint + "/" + transcript_id

    print("Transcribing at", polling_endpoint)
    return polling_endpoint, None

@st.cache_data
def get_analysis_results(polling_endpoint):

    status = 'submitted'

    while True:
        print(status)
        polling_response = requests.get(polling_endpoint, headers=headers)
        status = polling_response.json()['status']

        if status == 'submitted' or status == 'processing' or status == 'queued':
            print('not ready yet')
            sleep(10)

        elif status == 'completed':
            print('creating transcript')

            return polling_response

            break
        else:
            print('error')
            return False
            break

def clickable_images_with_titles(thumbnails, titles):
    selected = -1
    num_videos = len(thumbnails)
    num_rows = num_videos // 3 + (num_videos % 3 > 0)  # Calculate the number of rows needed
    for row in range(num_rows):
        with st.container():
            col1, col2, col3 = st.columns(3)  # Create three columns in each row
            for col, index in zip([col1, col2, col3], range(row * 3, min((row + 1) * 3, num_videos))):
                thumbnail = thumbnails[index]
                title = titles[index]
                col.image(thumbnail, caption='', use_column_width=True)
                if col.button(f'### {title}', key=f'button{index}', help=f'imagebutton{index}'):
                    selected = index
        st.markdown("<style> .stContainer { margin-bottom: 20px; } </style>", unsafe_allow_html=True)  # Add space between rows
    return selected

st.title("JuriSense 🎥", anchor=False)

st.markdown("#### What is JuriSense?")
st.markdown("JuriSense is a platform that implements Corrective Retrieval Augmented Generation (RAG) platform. It is designed to generate Video Notes, Summaries, and Questionnaires from YouTube videos and PDF documents, with a focus on the field of Criminal Law.")

# Highlighted Features
st.markdown("#### Why JuriSense is a Game-Changer?")
st.markdown("1. **Audio Generation:** Generate audio clips from the video contents.")
st.markdown("2. **Automated Transcript Generation:** 🤖 JuriSense uses advanced algorithms to accurately convert video content into text, saving you time and effort.")
st.markdown("3. **Comprehensive Video Summaries:** 📑 Dive deeper into the content with detailed summaries, allowing you to quickly grasp the essence of each video.")
st.markdown("4. **Questionnaire Generation:** 📝 Create questionnaires based on the video contents, making it easy to test your understanding.")

st.markdown("#### Key Features")
st.markdown("- **Efficient Study Material Creation:** ⏱️ JuriSense automates the process of generating study materials, saving time for students, researchers, educators, and professionals.")
st.markdown("- **Extensive Coverage:** 📚 Criminal Law is a complex field, encompassing statutes, regulations, precedents, and legal principles. JuriSense organizes this vast amount of information into concise and structured notes.")
st.markdown("- **Ethical Standards:** 🤝 JuriSense upholds ethical standards by providing accurate and reliable information, ensuring that users can trust the generated study materials.")
st.markdown("- **Accessibility:** 🌍 JuriSense is accessible to users across diverse contexts and jurisdictions, making it a valuable tool for anyone navigating legal content.")

# How It Works
st.markdown("#### How It Works:")
st.write("1. **Upload Video Links:** Upload a text file containing the YouTube video links of any channel you wish to assess.")
st.write("2. **Generate Insights:** Click on any thumbnail to gain access to a wealth of information, including video summaries, transcripts, and questionnaires regarding the selected video.")

# Call to Action
st.markdown("Ready to optimize your study materials? Try JuriSense today!")

default_bool = st.checkbox("Use a default file")

pdf_doc = "13law-of-crimes-I.pdf"
raw_text = get_pdf_text(pdf_doc)

# get the text chunks
text_chunks = get_text_chunks(raw_text)

# create vector store
vectorstore = get_vectorstore(text_chunks)
if default_bool:
    file = open("./law.txt")
else:
    file = st.file_uploader("Upload a file that includes the links (.txt)")

if file is not None:

    prompt = """Based on the following context items, please summarise the query as comprehensive and as in-depth as possible. 
Ensure that you cover everything important discussed in the video and provide as long of a summary as you need (I suggest around 1000 words if possible).
Make sure your answers are as explanatory as possible:

    """

    questionnaire_prompt = """Based on the following transcript, give five questions and answers. 
                            The answers should be very detailed and they should start from a new line
    """

    prompt_questionnaire = """Create 5 questions on the data from the transcript and 
                            generate their answers(in 100 words) as well from both the video's transcript content as well as the vectorstore. The answers should be very detailed and they should start from a new line.
    """

    dataframe = pd.read_csv(file, header=None)
    dataframe.columns = ['urls']
    urls_list = dataframe['urls'].tolist()

    titles = []
    locations = []
    thumbnails = []

    for video_url in urls_list:
        # download audio
        video_title, save_location, video_thumbnail = save_audio(video_url)
        if video_title:
            titles.append(video_title)
            locations.append(save_location)
            thumbnails.append(video_thumbnail)

    selected_video = clickable_images_with_titles(thumbnails, titles)

    if selected_video > -1:
        video_url = urls_list[selected_video]
        video_title = titles[selected_video]
        save_location = locations[selected_video]

        st.header(video_title)
        st.audio(save_location)

        # upload mp3 file to AssemblyAI
        audio_url, error = upload_to_AssemblyAI(save_location)
        
        if error:
            st.write(error)
        else:
            # start analysis of the file
            #polling_endpoint, error = start_analysis(audio_url)

            if error:
                st.write(error)
            else:
                # receive the results
                #results = get_analysis_results(polling_endpoint)
                transcript = transcriber.transcribe(audio_url, config)

                print(transcript.text)
                transcript_text = transcript.text

                summary, transcript_tab , questionnaire_tab= st.tabs(["Summary", "Transcript", "Questionnaire"])

                with transcript_tab:
                    st.header("Transcript of this Video: ")
                    st.write(transcript_text)

                with summary:
                    st.header("Summary of the Video: ")
                    summary = generate_gemini_content(transcript_text,prompt)
                    st.write(summary)
                    
                questionnaire = questionnaire_content(transcript_text, questionnaire_prompt, vectorstore)
                
                with questionnaire_tab:
                    st.header("Questionnaire:")
                    st.write(questionnaire)