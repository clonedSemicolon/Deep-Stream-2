import os
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from deepstreamhelper import helper
from imgprocessor import ImageProcessor
from streamlit_player import st_player


stream_helper = None
audio_bytes = None

def save_uploadedfile(uploadedfile):
     with open(os.path.join("dir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))


def build_helper(stHelper):
   if(stHelper.preprocess()):
    audio_file = open('./cloned_audio.wav', 'rb')
    audio_bytes = audio_file.read()
    

def print_helper():
    print("helper is changed")


def getCharacterName(ch):
    name = "Unknown"
    ch = str(ch)
    if(ch == "1"): 
        name = "May"
    elif (ch == "2"): 
        name = "McStay"
    elif (ch == "3"): 
        name = "Nadella"
    elif (ch == "4"): 
        name = "Obama1"
    elif (ch == "5"): 
        name = "Obama2"

    return name

st.title('Deep Stream')
st.subheader('This is a streamlit app for deep learning')


audioPath = "dir/"
textPath = "dir/"
isAudioUploaded = False
isTextUploaded = False
isUploaded = False

uploaded_audio_file = st.file_uploader("Choose an Audio File", type=[ "wav"],)
uploaded_text_file = st.file_uploader("Choose a Text File", type=["txt"])


if uploaded_audio_file is not None:
    audioPath += uploaded_audio_file.name
    save_uploadedfile(uploaded_audio_file)
    isAudioUploaded = True
    

if uploaded_text_file is not None:
    textPath += uploaded_text_file.name
    save_uploadedfile(uploaded_text_file)
    isTextUploaded = True
    

if isAudioUploaded and isTextUploaded:
    isUploaded = True
    stream_helper = helper(audioPath, textPath)
    stream_helper.loadFromBrowser()

if stream_helper is not None:
    st.button("Start Processing",
    on_click = build_helper,
    args = {stream_helper}
    )
    
if audio_bytes is not None:
    st.audio(audio_bytes, format='audio/wav')

character = st.selectbox(
    "Choose a Character",
    ["Character 1",
     "Character 2",
     "Character 3",
     "Character 4",
     "Character 5"]
    )

character = getCharacterName((character).split(" ")[1])

if (character is not None):
    imgProcessor = ImageProcessor(character)
    st.button("Sync Process",on_click = imgProcessor.startAudioSync)

print(character)


