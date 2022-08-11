import os

class spleeter():
    
    def spleeting_speech_and_music(path_audio_file: str, output_path: str):


        if os.path.isfile(path_audio_file):
            print('{path_audio_file}')
            cmd = 'spleeter separate -p spleeter:2stems -o output/ ' +  str(path_audio_file)
            print(cmd) 
            os.system(cmd)
            file_name = os.path.basename(os.path.normpath(path_audio_file))
            file_name = os.path.splitext(file_name)[0]
            directory = os.getcwd()
            audio_files = os.path.join(directory, output_path, file_name)
            music_file = os.path.join(audio_files, 'accompaniment.wav')
            speech_file = os.path.join(audio_files, 'vocals.wav')
            
            return music_file, speech_file
        else:
            None