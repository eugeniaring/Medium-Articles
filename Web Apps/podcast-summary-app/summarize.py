############ Summarize code #############

# https://www.youtube.com/watch?v=q-5uAFJGqOk&list=PLcWfeUsAys2lUkQ-s5wtyLhYPJ3B4qUK2&index=10
# https://www.listennotes.com/podcasts/naked-data-science-naked-data-science-_uJhei-AXdy/

import requests
from time import sleep
import json
import os
 
### Step 1: retrieve episode url from listen notes

f = open("secrets.json", "rb")
parameters = json.load(f)

url_episodes_endpoint = 'https://listen-api.listennotes.com/api/v2/episodes'
episode_id = "e9baa9118e654cd09baff7ac4b67228f"
headers = {
  'X-ListenAPI-Key': parameters["api_key_listennotes"],
}
url = f"{url_episodes_endpoint}/{episode_id}"
response = requests.request('GET', url, headers=headers)
print(response.json())
data = response.json()
audio_url = data['audio']

### Step 2: Transcribe and Summarize audio

api_key = parameters["api_key"]

'''autochapter feature provides a summary over time for audio content 
transcribed with AssemblyAI's API'''

transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

transcript_request = {
        'audio_url': audio_url,
        'auto_chapters': True
    }

headers = {
    "authorization": api_key,
    "content-type": "application/json"
}

transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
transcript_id = transcript_response.json()["id"]

### Step3: Save transcription and summary into 2 different files

polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
polling_response = requests.get(polling_endpoint, headers=headers)

while polling_response.json()["status"] != 'completed':
    sleep(5)
    polling_response = requests.get(polling_endpoint, headers=headers)

print('\njson file: ',polling_response.json())
## file 1 with transcription
filename = f"{transcript_id}.txt"
with open(filename, 'w') as f:
    f.write(polling_response.json()['text'])

# filename = transcript_id + '_chapters.json'
# ## file 2 with summary
# with open(filename, 'w') as f:
#     chapters = polling_response.json()['chapters']
#     json.dump(chapters, f, indent=4)


filename = transcript_id + '_chapters.txt'

## file 2 with summary
with open(filename, 'w') as f:
    chapters = polling_response.json()['chapters']
    for c in chapters:
       f.write(c['summary']+ '\n')
    json.dump(chapters, f, indent=4)
print('Transcript and Summary saved')
