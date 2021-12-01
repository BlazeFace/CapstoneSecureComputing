from rich import print
from rich.prompt import Prompt
import perturb
import threading
import PIL
from PIL import Image
import ffmpeg
import numpy as np

#TODO: 
# Traceback (most recent call last):
#   File "main.py", line 75, in <module>
#     setup()
#   File "main.py", line 36, in setup
#     probe = ffmpeg.probe(file_name)
#   File "/home/ugrads/majors/jamespur/securecomputing/CapstoneSecureComputing/.venv/lib64/python3.6/site-packages/ffmpeg/_probe.py", line 23, in probe
#     raise Error('ffprobe', out, err)
# ffmpeg._run.Error: ffprobe error (see stderr output for detail)
#
# When passing in filename that does not exist

class algoThread(threading.Thread):
    def __init__(self, threadID, algo, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.algo = algo
        self.files = files

    def run(self):
        print(f"Starting thread [bold red]{self.algo}[/bold red]")
        resp = perturb.evaluate(self.algo, self.files)
        print(f"{self.algo} finished with a score of {resp}")


def setup():
    print("Welcome to [bold green]Truth[/bold green]!")
    mode = Prompt.ask("Please select mode", choices=['Local Preturb', 'Remote Preturb', 'Evaluate', 'Break Video'])
    # Some amount of choices
    if mode == 'Local Preturb':
        files_raw_names = Prompt.ask('Enter Filenames comma seperated')
        file_names = files_raw_names.split(",")
        files = [file_name.strip() for file_name in file_names]
        local_evaluate(files)
    elif mode == 'Break Video':
        file_raw_name = Prompt.ask('Enter Filename')
        file_name = file_raw_name.strip()
        probe = ffmpeg.probe(file_name)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        num_frames = int(video_info['nb_frames'])

        out, err = (
            ffmpeg
                    .input(file_name)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True)
            )
        video = (
                np
                    .frombuffer(out, np.uint8)
                    .reshape([-1, height, width, 3])
            )
        print(video.shape)
        print(num_frames)
        print(height)
        print(width)
        
        #TODO: FIX THIS, TEMPORARY CALL
        perturb.evaluate("torchattacks_facenet_vggface2", video)
        
    elif mode == 'List Algo':
        print(perturb.methods())
    else:
        print('Failed to Setup')


def local_evaluate(filenames):
    files = [PIL.Image.open(file) for file in filenames]
    algorithms = perturb.methods()
    threads = [algoThread(i, algo, files) for i, algo in enumerate(algorithms)]
    [thread.start() for thread in threads]


if __name__ == "__main__":
    setup()
