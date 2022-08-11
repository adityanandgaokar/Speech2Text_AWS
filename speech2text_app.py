import streamlit as st 
import io 
import os 
from PIL import Image

from speech2text import Speech2Text
from spleeter import spleeter

from streamlit_utils import get_binary_file_downloader_html, df2excel
from Melspectrogram import melspectrogram
from audio_tagger import Audio_Tagger
from PANNs import PANNs
import numpy as np
from panns_inference import labels

left_col, middle_col, right_col = st.columns((3, 1, 5))

#path to save uploaded file in local machine
wav_files_path = "C:/Users/AdityaNandgaokar/Aimpower/audio_files_path_s2t/"

#path of PANNs tagger checkpoint Cnn14_mAP=0.431.pth or Wavegram_Logmel_Cnn14_mAP=0.439.pth 
audio_tagger = 'C:/Users/AdityaNandgaokar/Aimpower/panns/Cnn14_mAP=0.431.pth'

#path to save melspectrogram results in local machine 
mel_spectrogram_path = 'C:/Users/AdityaNandgaokar/Aimpower/mel_spectrogram'

#path to save spleeter results
output = 'output'
assembly_languages =["en"]
aws_languages = ["en-GB", 'hi-IN']
available_languages = assembly_languages + aws_languages  

uploaded_file = st.sidebar.file_uploader("upload a movie", type=["mp4", "mov", "gif", 'wav', 'mp3'])
lang = st.sidebar.selectbox("asset language", available_languages, index=available_languages.index("en"))
brand_name = st.sidebar.text_input('Brand Name')
product_name = st.sidebar.text_input('Product Name')
brand_name = None if brand_name =="" else brand_name 
product_name = None if product_name =="" else product_name 

if lang in assembly_languages: 
    s2t_api = "assembly"
else: 
    s2t_api = "aws"

s2t = Speech2Text(s2t_api, lang)
mel_spec = melspectrogram()
aud_tag = Audio_Tagger()
panns = PANNs(audio_tagger)
run = st.sidebar.button("run speech2text")

if run: 
   
    if uploaded_file is not None: 
        
        extension = uploaded_file.type.rsplit('/', 1)[-1]    
        print(extension)
        if extension in ['mp4', 'mov', 'gif']:
            print(uploaded_file.type.rsplit('/', 1)[-1])
            print('haha')
            g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
            name = uploaded_file.name
            video_path = os.path.join("D:/Aimpower/MP4_files", name)
            ########## SAVE VIDEO BYTES 
            with open(video_path, 'wb') as out:  ## Open temporary file as bytes
                video_bytes = g.read()
                out.write(video_bytes)   

            with left_col: 
                st.video(video_bytes)

              
            wav_file_path, fps = s2t.video2wav(video_path, wav_files_path)
            
        elif extension in ['mp3', 'wav', 'mpeg']:

            g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
            name = uploaded_file.name
            wav_file_path = os.path.join(wav_files_path, name)
            ########## AUDIO BYTES 
            with open(wav_file_path, 'wb') as out:  ## Open temporary file as bytes
                audio_bytes = g.read()
                out.write(audio_bytes)   
            print(wav_file_path)
            # as we uploaded audio we dont have fps for it.
            fps = 22
        


        if not wav_file_path is None: 
            
            with right_col:
                    
                st.audio(wav_file_path)  
                st.markdown(get_binary_file_downloader_html(wav_file_path, 'WAV'), unsafe_allow_html=True)
                
                # separating music and speech 
                music_file, speech_file = spleeter.spleeting_speech_and_music(wav_file_path, output)

                # music dataframe and tag with segmentation of start and end time information
                tag, dataframe = Audio_Tagger.audio_timestamps(music_file, 'music')
                if tag  == None:
                    st.write("Music does not exist in {} file".format(wav_file_path))
                else:    

                    st.write("Audio file has: ", tag)
                    st.write("start and end time of {} segments".format(tag), dataframe.astype(str))
                    st.markdown(df2excel(dataframe, "", name + "{} Timestamps".format(tag)), unsafe_allow_html=True) 
                
                # speech dataframe and tag with segmentation of start and end time information
                tag, dataframe = Audio_Tagger.audio_timestamps(speech_file, 'speech')
                if tag == None:
                    st.write("Speech does not exist in {} file".format(wav_file_path))
                else:
                    st.write("Audio file has: ", tag)
                    st.write("start and end time of {} segments".format(tag), dataframe.astype(str))
                    st.markdown(df2excel(dataframe, "", name + "{} Timestamps".format(tag)), unsafe_allow_html=True) 
                    

                with st.spinner("transcribing speech -> text"): 
                    print(wav_file_path)
                    if s2t_api == "assembly":
                        transcribed_text, s2t_df, json = s2t.run_speech2text(wav_file_path, brand_name, product_name, fps) 
                    elif s2t_api == "aws":
                        transcribed_text, s2t_df, s2t_df2, json = s2t.run_speech2text(wav_file_path, brand_name, product_name, fps)
                    st.write("text: ", transcribed_text)
                    s2t_df = s2t.add_frame_info(s2t_df, fps)
                    s2t_df = s2t.brand_product_detection(s2t_df, brand_name, product_name)
                    print(s2t_df)
                    st.write("word by word and timing", s2t_df.astype(str))
                    st.markdown(df2excel(s2t_df, "", name + "speech2text_results"), unsafe_allow_html=True) 
                        
                    if s2t_api == 'aws':
                        st.write("start and end time of speaker information", s2t_df2.astype(str))
                        st.markdown(df2excel(s2t_df2, "", name + "speech2text_results"), unsafe_allow_html=True) 

                    tags = panns.top_tags_audio_file(wav_file_path)
                    sorted_indexes = np.argsort(tags[0])[::-1]

                    st.write('Audio Classification (Top Audio Tags)')
                    
                    for k in range(10):
                        print('{}: {: .3f}'.format(np.array(labels)[sorted_indexes[k]], tags[0][sorted_indexes[k]]))
                        st.write('{}: {: .3f}'.format(np.array(labels)[sorted_indexes[k]], tags[0][sorted_indexes[k]]))
                         

                    mel_image_path = mel_spec.get_melspectrogram(wav_file_path, mel_spectrogram_path)
                    mel_image = Image.open(mel_image_path)
                    st.image(mel_image)
                    #emotion = s2t.get_emotion(wav_file_path)
                    #s2t.write('Emotion', emotion)
                    st.write("full result", json)
                        
                    try: video_path
                    except NameError: video_path = None 

                    if video_path == None:
                        s2t.upload_to_s3(wav_file_path, transcribed_text, s2t_df, json, None)
                    else:
                        s2t.upload_to_s3(wav_file_path, transcribed_text, s2t_df, json, video_path)
                            

                    
        else: 
            st.write("could not extract wav from video")
