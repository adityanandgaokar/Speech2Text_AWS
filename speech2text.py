import os
from random import sample
from matplotlib.pyplot import text, title 
import requests 
import moviepy.editor as mp
import pandas as pd 
import cv2
from fuzzywuzzy import fuzz
import boto3
from io import StringIO
import json
import names
import time
import urllib
import sys
from subprocess import check_output
import json
from scipy.io import wavfile
import numpy as np

sys.path.append(os.path.abspath(f'D:/Aimpower/Main/Speech2Text_Assembly/'))

# from utils.transforms import ToTensor1D

# from model import AudioCLIP


class Speech2Text():
    def __init__(self, API: str, lang: str = "en", verbose: bool=True):  
        self.lang = lang
        self.API = API.lower()
        if API.lower() == "assembly": 
            self.API_key = "api key of assembly ai"  
        elif API.lower() == "aws": 
            self.API_ID = 'id of aws api'
            self.API_key = 'key of aws api'
        
    def run_speech2text(self, wav_file_path, brand_name: str, product_name: str, fps: int): 
        
        if self.API == "assembly": 
            transcribed_text, s2t_df, assembly_json =  self.run_assembly(wav_file_path, brand_name, product_name)

            return transcribed_text, s2t_df, assembly_json
        elif self.API == "aws":
        
            transcribed_text, aws_df, aws_df2, aws_json = self.run_aws( wav_file_path, brand_name, product_name)
            
            return transcribed_text, aws_df, aws_df2, aws_json


    def run_assembly(self, wav_file_path, brand_name: str, product_name: str):                
        headers = {
                'authorization': self.API_key,
                'content-type': 'application/json'
                }
        
        transcript_endpoint = 'https://api.assemblyai.com/v2/transcript'
        upload_endpoint = 'https://api.assemblyai.com/v2/upload'

        def _read_wav_file(wav_file_path, chunk_size=5242880):
            with open(wav_file_path, 'rb') as _file:
                while True:
                    data = _file.read(chunk_size)
                    if not data:
                        break
                    yield data

        upload_response = requests.post(upload_endpoint, headers= headers, data= _read_wav_file(wav_file_path))                                     
        word_boost = [brand_name, product_name]
        word_boost = [w for w in word_boost if w is not None ]

        if len(word_boost) > 0:                     
            transcript_request = {'audio_url': upload_response.json()['upload_url'],
                                'word_boost': word_boost,
                                'boost_param': 'high',
                                'speaker_labels': True}
        
        else:                    
            transcript_request = {'audio_url': upload_response.json()['upload_url']}
        
        transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers = headers)
        pooling_response = requests.get(transcript_endpoint+ '/' + transcript_response.json()['id'], headers=headers)
        
        while pooling_response.json()['status'] != 'completed':
            pooling_response = requests.get(transcript_endpoint+ '/' + transcript_response.json()['id'], headers=headers)
        
        assembly_json = pooling_response.json()
        return assembly_json["text"], pd.DataFrame(assembly_json["words"]).rename(columns={"end": "end_time"}), assembly_json


    def run_aws(self, wav_file_path, brand_name: str, product_name: str):

        print(self.API_ID)
        s3 = boto3.client('s3',
                 aws_access_key_id = self.API_ID,
                 aws_secret_access_key = self.API_key,
                 region_name = 'eu-central-1') 

        folder_name = os.path.basename(os.path.normpath(wav_file_path))
        folder_name = os.path.splitext(folder_name)[0]



        s3.upload_file(wav_file_path, 'spectral-analysis', folder_name + '/' + os.path.basename(os.path.normpath(wav_file_path)))

        word_boost = [brand_name, product_name]
        word_boost = [word for word in word_boost if word is not None]

        trascribe_client = boto3.client('transcribe',
                                        aws_access_key_id = self.API_ID,
                                        aws_secret_access_key = self.API_key,
                                        region_name = 'eu-central-1'               
                                        )
                                    

        file_uri = 's3://' + 'spectral-analysis' +  '/' + folder_name + '/' +  os.path.basename(os.path.normpath(wav_file_path))

        job_name = names.get_last_name()

        job_name_vocab = names.get_first_name()
        print(self.lang)

        response_vocab = trascribe_client.create_vocabulary(VocabularyName = job_name_vocab,
                                           LanguageCode = self.lang,
                                           Phrases=word_boost)
        
        
        time.sleep(80)
        print(response_vocab['VocabularyState'])
        trascribe_client.start_transcription_job(
                                                            TranscriptionJobName = 'rekha',  
                                                            Media={'MediaFileUri': file_uri},
                                                            MediaFormat= 'wav',
                                                            Settings= {'VocabularyName': job_name_vocab,
                                                            'ShowSpeakerLabels': True,
                                                            'MaxSpeakerLabels' : 3},
                                                            LanguageCode = self.lang
                                                    
        )     
        print(response_vocab['VocabularyState'])


        max_tries = 60
        while max_tries > 0:
            max_tries -= 1
            

            aws_json = trascribe_client.get_transcription_job(TranscriptionJobName = 'rekha')
            job_status = aws_json['TranscriptionJob']['TranscriptionJobStatus']

            if job_status in ['COMPLETED', 'FAILED']:
                print(f'{job_status}.')
                if job_status == 'COMPLETED':
                    response = urllib.request.urlopen(aws_json['TranscriptionJob']['Transcript']['TranscriptFileUri'])
                    data = json.loads(response.read())
                    text = data['results']['transcripts'][0]['transcript']
                    
                break
            else:
                print(f'current status is {job_status}.')
            time.sleep(10)

        print(data)
        df1 = pd.DataFrame(data['results']['items'])
        df2 = pd.DataFrame(data['results']['speaker_labels']['segments'])
        
        df1['confidence'] = ''
        df1['text'] = ''
        
        for i in range(len(df1['alternatives'])):
            df1['confidence'][i] = df1['alternatives'][i][0]['confidence'] 

        for j in range(len(df1['alternatives'])):
            df1['text'][j] = df1['alternatives'][j][0]['content'] 

        df1 = df1.drop('alternatives', 1)

        df2 = df2.drop(['items'], axis = 1)



        return text, df1, df2, data






    def add_frame_info(self, s2t_df: pd.DataFrame, fps: int):
        if self.API == 'assembly': 
            s2t_df['start_frame'] = s2t_df['start'].apply(lambda start: int((start/1000) * fps))
            s2t_df['end_frame'] = s2t_df['end_time'].apply(lambda end_time: int((end_time/1000) * fps))
            return s2t_df
        elif self.API == 'aws':

            s2t_df['start_time'] = s2t_df['start_time'].fillna(0)             
            s2t_df['end_time'] = s2t_df['end_time'].fillna(0)

            s2t_df['start_frame'] = s2t_df['start_time'].apply(lambda start: int(float(start) * fps))
            s2t_df['end_frame'] = s2t_df['end_time'].apply(lambda end: int(float(end) * fps))

            return s2t_df

    def brand_product_detection(self, s2t_df: pd.DataFrame, brand_name: str, product_name: str):
        s2t_df['has_brand'] = s2t_df['text'].apply(lambda text: fuzz.token_set_ratio(text, brand_name))
        s2t_df['has_product'] = s2t_df['text'].apply(lambda text: fuzz.token_set_ratio(text, product_name))
        
        return s2t_df
    

    def video2wav(self, video_path, out_dir):            
        try: 
            
            #for video in videos 
            clip = mp.VideoFileClip(video_path)
            file_name = os.path.basename(os.path.normpath(video_path))
            file_name = os.path.splitext(file_name)[0].strip()
            wav_file_path = os.path.join(out_dir, file_name + '.wav')
            if clip.audio == None:
                return None, None
            else:
                clip.audio.write_audiofile(wav_file_path)
                
                capture = cv2.VideoCapture(video_path)  # open the video using OpenCV
                fps = int(capture.get(cv2.CAP_PROP_FPS))
                print(fps)
                return wav_file_path, fps 
        except: 
            return None
            
    def upload_to_s3(self, wav_file_path: str, transcribed_text: str, s2t_df: pd.DataFrame, assembly_json, video_path: str):
        
    
        session = boto3.Session(aws_access_key_id=self.API_ID,
                                aws_secret_access_key=self.API_key
                                )

        s3 = session.resource('s3')
        my_bucket = 'spectral-analysis'

        folder_name = os.path.basename(os.path.normpath(wav_file_path))
        folder_name = os.path.splitext(folder_name)[0] 

        if self.API == 'assembly':
            s3.meta.client.upload_file(wav_file_path, my_bucket, folder_name + '/'  +  os.path.basename(os.path.normpath(wav_file_path)))
        
        if video_path != None:
            s3.meta.client.upload_file(video_path, my_bucket, folder_name + '/' +  os.path.basename(os.path.normpath(video_path)))

        file_name = os.path.basename(os.path.normpath(wav_file_path))
        file_name = os.path.splitext(file_name)[0].strip()

        s3_txt = s3.Object(my_bucket, folder_name + '/' +  file_name + '.txt' )
        response_txt = s3_txt.put(Body = transcribed_text)

        csv_buffer = StringIO()
        s2t_df.to_csv(csv_buffer)

        s3_df = s3.Object(my_bucket, folder_name + '/' +  file_name + '.csv').put(Body=csv_buffer.getvalue())
        
        s3_json = s3.Object(my_bucket, folder_name + '/' +  file_name + '.json')
        response_vid = s3_json.put(Body=(bytes(json.dumps(assembly_json).encode('UTF-8'))))

    
    
    def has_audio_or_not(self, wav_file_path: str):
        
        sil_threshold=0.03
        win_size=0.25

        samplerate, data = wavfile.read(wav_file_path)

        #holds length of numpy array
        len_data = len(data)

        audio_duration = float(len_data / samplerate)

        print(audio_duration)


        win_frames = int(samplerate * win_size) # number of samples in a window
        win_amps = [] # windows in which to measure amplitude
        for win_start in np.arange(0, len(data), win_frames):
            # Find the end of the window
            win_end = min(win_start + win_frames, len(data))
            # Add the mean amplitude for this frame to the list of window amplitudes
            win_amps.append(np.nanmean(np.abs(data[win_start:win_end])))

        # Calculate the minimum threshold for a window to be silent
        threshold = sil_threshold * max(win_amps)

        # Find the windows that are silent
        sils, = np.where(win_amps <= threshold)

        # The silence length is the number of silent windows times the window length
        sil_dur = float(len(sils) * win_size)

        print(sil_dur)

        return audio_duration, sil_dur
