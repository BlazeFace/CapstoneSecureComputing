How to use tool
FFMpeg must be installed, it can be found at [FFMpeg](https://ffmpeg.org/download.html).
Python 3.6 or 3.7 must be used.

1. Download the zip or git clone
2. run pip install -r requirements.txt
3. run main script with python3 main.py
4. To premute video use the "Break Video" option then enter the name of the video
5. Your video will be processed and then evaluations will be returned along with the video output
How permutations work

How to generate eval
1. Once you have all the images you from the video being processed you can call the imagecomp.py
class, which will result in the average SSIM value being returned allowing you to tell how similar
the perturbed and unperturbed images look.
etc
