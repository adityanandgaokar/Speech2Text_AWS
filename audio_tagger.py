
from re import match
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from torch import chunk
import pandas as pd


class Audio_Tagger():
    
    def audio_exist_or_not(wav_file_path, label):
        
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

        # The silence length is the number of silent windows times the window length (silence in seconds)
        sil = float(len(sils) * win_size)


        if audio_duration == sil:
            return 'silence'
        else:
            return label
        



    # segments of music / speech with start and end time in the form of dataframe    
    def audio_timestamps(wav_file_path, label):  
        
        # check whether speech / music present in audio or not 
        tag = Audio_Tagger.audio_exist_or_not(wav_file_path, label)

            
        if tag != 'silence': 
            target_dBFS = -20.0
            # converting wav to audio_segment
            audio_seg = AudioSegment.from_wav(wav_file_path)

            # normalized audio_segment to -20dBFS and adjust target amplitude 
            change_in_dBFS = target_dBFS - audio_seg.dBFS
            normalized_audio = audio_seg.apply_gain(change_in_dBFS)


            # detected non-silent chunks
            nonsilent_data = detect_nonsilent(normalized_audio, min_silence_len=500, silence_thresh=-30, seek_step=1)
            df = pd.DataFrame(columns=['start_time', 'end_time'])
            i = 0
            for chunks in nonsilent_data:
                timestamps = [chunk/1000 for chunk in chunks]
                print(tag)
                print(timestamps)
                df.loc[i] = [timestamps[0], timestamps[1]]
                i = i + 1 
                
            return tag, df
        elif tag == 'silence':
            return None, None




