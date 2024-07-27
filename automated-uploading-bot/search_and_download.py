from youtube_search import YoutubeSearch
from pytubefix import YouTube
from pytubefix.cli import on_progress
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
import os

results = YoutubeSearch('football',max_results = 10).to_dict()
youtube_path = "https://www.youtube.com/watch?v="
process_dir = "process_dir"
split_dir = "process_split_dir"
if os.path.isdir(process_dir):
    os.mkdir(process_dir)
if os.path.isdir(split_dir):
    os.mkdir(split_dir)

video_ids = []
video_path = []


for video in results:
    url = youtube_path + video['id']
    print(video)
    yt = YouTube(url, on_progress_callback = on_progress)
    print(yt.title)
    if str(video['duration']).isnumeric():
        pass
    elif int(video['duration'].split(":")[1]) > 0:
        if video['url_suffix'].split("/")[1] == 'shorts':
            pass
        else:
            ys = yt.streams.get_highest_resolution()
            ys.download(process_dir)  


for path in os.listdir(process_dir):
    scene_list = detect(os.path.join(process_dir,path), AdaptiveDetector())
    video_name = path.split(".")[0]
    if not os.path.isdir(os.path.join(split_dir,video_name)):
        os.mkdir(os.path.join(split_dir,video_name))
    split_video_ffmpeg(os.path.join(process_dir,path), scene_list,output_dir=os.path.join(split_dir,video_name))