import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import numpy as np

class melspectrogram():

    def get_melspectrogram(self, wav_file_path: str, mel_spectrogram_path: str):

            if wav_file_path is not None:
                signal, sample_rate = librosa.load(wav_file_path)
                mel_spec = librosa.feature.melspectrogram(y= signal, sr=sample_rate)
                fig, ax = plt.subplots()
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                image = librosa.display.specshow(mel_spec_db, x_axis = 'time', y_axis = 'mel',
                                                sr= sample_rate, ax=ax)
                fig.colorbar(image, ax=ax, format='%+2.0f dB')
                ax.set(title='Mel-frequency spectrogram')

                mel_image_path = mel_spectrogram_path + '/' +os.path.basename(os.path.normpath(wav_file_path)) + '.png'
                plt.savefig(mel_image_path)
                return mel_image_path
            else:
                None



