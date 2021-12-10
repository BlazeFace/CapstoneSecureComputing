from rich import print
from rich.prompt import Prompt
import perturb
import ffmpeg
import numpy as np


def setup():
    print("Welcome to [bold green]Truth[/bold green]!")
    mode = Prompt.ask("Please select mode", choices=['Local Perturb', 'Remote Perturb', 'Evaluate', 'Break Video', 't'])
    # Some amount of choices
    if mode == 'Local Perturb':
        files_raw_names = Prompt.ask('Enter Filenames comma seperated')
        file_names = files_raw_names.split(",")
        files = [file_name.strip() for file_name in file_names]
        local_evaluate(files)
    elif mode == 'Break Video' or mode == "t":  # temporary test mode
        if mode == "t":
            file_raw_name = "input/samplevid.mp4"
        else:
            file_raw_name = Prompt.ask('Enter Filename')
        file_name = file_raw_name.strip()
        try:
            probe = ffmpeg.probe(file_name)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            num_frames = int(video_info['nb_frames'])
        except:
            print("Failed to Find File")
            return
        # Uses FFMPEG to break video down to numpy arrays
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
        # Using the perturb API structure we call our attack
        perturb.evaluate("torchattacks_facenet_casiawebface", video)
    elif mode == 'List Algo':
        print(perturb.methods())
    else:
        print('Failed to Setup')


def local_evaluate(filenames):
    # TODO Need to enable it
    print("Not Yet Implemented")


if __name__ == "__main__":
    setup()
