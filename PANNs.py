
import librosa
from matplotlib.pyplot import savefig
from panns_inference import AudioTagging
import torch
import numpy as np

class PANNs():
    def __init__(self, tagger_path: str):
        #path of tagger checkpoint Cnn14_mAP=0.431.pth or Wavegram_Logmel_Cnn14_mAP=0.439.pth
        self.tagger = tagger_path

    def top_tags_audio_file(self, audio_path: str):         

        (audio, _) = librosa.core.load(audio_path, sr=32000, mono =True)
        audio = audio[None, :]

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device='cpu'

        print('------ Tags ------')
        at = AudioTagging(checkpoint_path=self.tagger , device='cuda')
        (tags, embedding) = at.inference(audio)
        
        return tags
        #print audio tagging top probabilities
        
