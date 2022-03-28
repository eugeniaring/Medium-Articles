#pip install SpeechRecognition
#pip install moviepy

import speech_recognition as sr 
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

### 4: Cut video file into chunks of 1 minute and convert each chunk into text format


# the video is 52 minutes long
num_seconds_video= 52*60
print("The video is {} seconds".format(num_seconds_video))
l=list(range(0,num_seconds_video+1,60))
print(l)

diz={}
for i in range(len(l)-1):
    ffmpeg_extract_subclip("videorl.mp4", l[i]-2*(l[i]!=0), l[i+1], targetname="chunks/cut{}.mp4".format(i+1))
    clip = mp.VideoFileClip(r"chunks/cut{}.mp4".format(i+1)) 
    clip.audio.write_audiofile(r"converted/converted{}.wav".format(i+1))
    r = sr.Recognizer()
    audio = sr.AudioFile("converted/converted{}.wav".format(i+1))
    with audio as source:
      r.adjust_for_ambient_noise(source)  
      audio_file = r.record(source)
    result = r.recognize_google(audio_file)
    diz['chunk{}'.format(i+1)]=result
    
### 5: Export results into a Text document


l_chunks=[diz['chunk{}'.format(i+1)] for i in range(len(diz))]
text='\n'.join(l_chunks)

with open('recognized.txt',mode ='w') as file: 
   file.write("Recognized Speech:") 
   file.write("\n") 
   file.write(text) 
   print("ready!")
