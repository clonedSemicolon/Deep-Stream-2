import subprocess
class ImageProcessor:

    def __init__(self,char):

        self.audioPath = "cloned_audio.wav"
        self.character = str(char)
        self.processor = "cpu"

    
    def startAudioSync(self):
        self.command = "python demo.py"
        self.command += " --id {}".format(self.character)
        self.command += "  --driving_audio ./{}".format(self.audioPath)
        self.command += " --device {}".format(self.processor)
        self.process()


    
    def process(self):
        try:
            subprocess.run(self.command, shell=True)
        except Exception as e:
            print(e)





# demo.py --id May  --driving_audio ./data/Input/00083.wav --device cpu
# demo.py --id May  --driving_audio ./cloned_audio.wav --device cpu



