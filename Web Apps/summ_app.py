import streamlit as st
import requests
import zipfile 
import json
from time import sleep

def retrieve_url_podcast(parameters,episode_id):
    url_episodes_endpoint = 'https://listen-api.listennotes.com/api/v2/episodes'
    headers = {
    'X-ListenAPI-Key': parameters["api_key_listennotes"],
    }
    url = f"{url_episodes_endpoint}/{episode_id}"
    response = requests.request('GET', url, headers=headers)
    print(response.json())
    data = response.json()
    audio_url = data['audio']
    return audio_url

def send_transc_request(headers,api_key,audio_url):
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    transcript_request = {
            'audio_url': audio_url,
            'auto_chapters': True
        }
    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
    transcript_id = transcript_response.json()["id"]   
    return transcript_id 

def obtain_polling_response(headers,transcript_id):
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    polling_response = requests.get(polling_endpoint, headers=headers)
    i=0
    while polling_response.json()["status"] != 'completed':
        sleep(5)
        polling_response = requests.get(polling_endpoint, headers=headers)
    return polling_response

def save_files(transcription,summary):

    with open('transcript.txt', 'w') as f:
        f.write(transcription)
        f.close()
    with open('summary.txt', 'w') as f:
        f.write(summary)
        f.close()    
    list_files = ['transcript.txt','summary.txt']
    with zipfile.ZipFile('final.zip', 'w') as zipF:
      for file in list_files:
         zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)
      zipF.close()

# title of web app

st.markdown('# **Summarizer App**')
bar = st.progress(0)

st.sidebar.header('Input parameter')

with st.sidebar.form(key='my_form'):
    episode_id = st.text_input('Insert Episode ID:')
    #bbc965b98747439abf0fe5c1a5ddfe5c
    #e9baa9118e654cd09baff7ac4b67228f
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    f = open("secrets.json", "rb")
    parameters = json.load(f)
    # step 1 - Extract episode's url from listen notes
    audio_url = retrieve_url_podcast(parameters,episode_id)
    #bar.progress(30)
    api_key = parameters["api_key"]
    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    # step 2 - retrieve id of transcription response from AssemblyAI
    transcript_id = send_transc_request(headers,api_key,audio_url)
    #bar.progress(70)

    # step 3 - transcription and summary
    polling_response = obtain_polling_response(headers,transcript_id)
    transcription = polling_response.json()["text"]
    chapters = polling_response.json()['chapters']
    summary = [c['summary'] for c in chapters]
    summary = '. '.join(summary)

    #bar.progress(100)
    st.header('Transcription')
    st.success(transcription)
    st.header('Summary')
    st.success(summary)
    save_files(transcription,summary)

    with open("final.zip", "rb") as zip_download:
        btn = st.download_button(
            label="Download",
            data=zip_download,
            file_name="final.zip",
            mime="application/zip"
        )
