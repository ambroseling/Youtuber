
import os
import numpy as np
import srt
import re
import ffmpeg
import whisperx
from whisperx.utils import get_writer
import gc
import soundfile as sf
from scipy.io.wavfile import write
from groq import Groq
from transformers import BarkModel,AutoProcessor
import torch
from dotenv import load_dotenv, find_dotenv
from diffusers import StableDiffusion3Pipeline
import os
import subprocess
import nvidia.cublas.lib
import nvidia.cudnn.lib

print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))

def create_video(image_file, audio_file, subtitle_file, output_file):
    # FFmpeg command to combine image, audio, and subtitles into a video
    command = [
        'ffmpeg',
        '-loop', '1',                   # Loop the image
        '-i', image_file,               # Input image
        '-i', audio_file,               # Input audio
        '-vf', f"ass={subtitle_file}",  # Video filter for ASS subtitles
        '-c:v', 'libx264',              # Video codec
        '-c:a', 'aac',                  # Audio codec
        '-pix_fmt', 'yuv420p',          # Pixel format
        '-b:a', '192k',                 # Audio bitrate
        '-movflags', 'faststart',       # Optimize for streaming
        '-shortest',                    # Match the shortest input duration
        '-f', 'mp4',                    # Output format
        output_file                     # Output file
    ]

    # Run the command
    subprocess.run(command, check=True)

def update_ass_styles(input_file, output_file, new_styles):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Define the pattern to find the [V4+ Styles] section
    style_pattern = re.compile(r'^Style:.*$')

    with open(output_file, 'w', encoding='utf-8') as file:
        in_styles_section = False
        for line in lines:
            if line.startswith('[V4+ Styles]'):
                in_styles_section = True
                file.write(line)
                continue
            if in_styles_section and line.startswith('['):
                in_styles_section = False

            if in_styles_section and style_pattern.match(line):
                style_name = line.split(',')[0].split(':')[1].strip()
                if style_name in new_styles:
                    style_parts = line.strip().split(',')
                    for key, value in new_styles[style_name].items():
                        style_parts[key] = value
                    new_line = ','.join(style_parts) + '\n'
                    file.write(new_line)
                else:
                    file.write(line)
            else:
                file.write(line)

def remove_overlapping_subtitles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cleaned_lines = []
    subtitle_pattern = re.compile(r"Dialogue:.*")
    timestamps = set()

    for line in lines:
        match = subtitle_pattern.match(line)
        if match:
            parts = line.split(',')
            start_time = parts[1].strip()
            end_time = parts[2].strip()

            if start_time in timestamps:
                continue
            timestamps.add(start_time)

        cleaned_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_lines)

def main(concept):
    image_file = 'albert.png'               # Path to your image file
    audio_file = 'example.wav'               # Path to your audio file
    subtitle_file = 'example.ass'        # Path to your ASS subtitle file
    output_file = f'{concept.replace(" ","_")}.mp4'       # Desired output video file path

    load_dotenv(".env")
    api_key  = os.getenv("APIKEY")
    client = Groq(
        api_key=api_key,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark-small").to(device)
    whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    voice_preset = "v2/en_speaker_6"
    concept = concept
    chat_completion_0 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Teach me how to write {concept} in python. Put the code snippet specifically in brackets. Please only provide 1 code snippet. Please fill in this template pretending you are explaining this code snippet to someone (DO NOT MODIFY ANYWHERE ELSE IN THE TEMPLATE EXCEPT SQUARE BRACKETS):\
                Here is how to write [INSERT CONCEPT (DO NOT PUT SHORT FORM FULL NAME ONLY)]:\
                ``` [PUT FULL CODE SNIPPET HERE, NOWHERE ELSE, DO NOT PUT CODE BELOW]``` \
                [00:00-00:05] [INSERT SCRIPT THAT EXPLAINS THE FIRST 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS EXTREMELY SHORT, LESS THAN 15 WORDS]\
                [00:05-00:10] [INSERT SCRIPT THAT EXPLAINS THE NEXT 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS EXTREMELY SHORT, LESS THAN 15 WORDS]\
                [00:10-00:15] [INSERT SCRIPT THAT EXPLAINS THE NEXT 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS EXTREMELY SHORT, LESS THAN 15 WORDS]\
                [00:15-00:20] [INSERT SCRIPT THAT EXPLAINS THE NEXT 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS EXTREMELY SHORT, LESS THAN 15 WORDS]\
                [00:20-00:25] [INSERT SCRIPT THAT EXPLAINS THE NEXT 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS EXTREMELY SHORT, LESS THAN 15 WORDS]\
                [00:25-00:30] [INSERT SCRIPT THAT EXPLAINS THE LAST 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS EXTREMELY SHORT, LESS THAN 15 WORDS]\            ",
            },
        ],
        model="llama3-8b-8192",
    )

    chat_completion_1 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Summarize this concept in 30 words in full sentences: {concept}",
            }
    ,
        ],
        model="llama3-8b-8192",
    )

    # import ipdb; ipdb.set_trace()

    content = chat_completion_0.choices[0].message.content
    summary =  chat_completion_1.choices[0].message.content
    print(content)
    print(f"Summary: {summary}")
    content = content.split("```")
    intro , code,scripts= content[0], content[1],content[-1]

    intro = intro.replace("\n","")

    scripts = scripts.split("\n")
    scripts = [x for x in scripts if x]
    script_wtimes = {line.split("]")[0]+"]": line.split("]")[1][1:]  for line in scripts}

    full_audio = np.zeros((1,1))
    sampling_rate = 24000
    i = 0
    text = ""

    for line in script_wtimes:
        l = script_wtimes[line]
        text += l
        inputs = processor(l, voice_preset=voice_preset).to(device)

        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy()
        full_audio = np.hstack((full_audio,audio_array))

    # import ipdb; ipdb.set_trace()
    write("example.wav",sampling_rate,  data = (full_audio.T*32767).astype("int16"))

    if os.path.exists("example.wav"):
        audio = whisperx.load_audio("example.wav")
        result = whisper_model.transcribe(audio, batch_size=batch_size)
        language_code=result["language"]
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        result['language'] = language_code
        vtt_writer = get_writer("srt", "captions/")
        vtt_writer(
            result,
            "example.wav",
            {"max_line_width": 15, "max_line_count": 1, "highlight_words": True},
        )
        (ffmpeg.input("captions/example.srt").output('example.ass').run())

        # Usage
        ass_name = str("example.ass")
        # we define the styles here:
        new_styles = {
            "Default": {
                1: "Arial",             # Fontname
                2: "12",                # Fontsize
                3: "&H00FF00FF",        # PrimaryColour (green)
                4: "&H000000FF",        # SecondaryColour
                5: "&H00000000",        # OutlineColour
                6: "&H64000000",        # BackColour
                16: "2",                # Outline (thickness)
                17: "1"                 # Shadow (depth)
            }
        }
        update_ass_styles(subtitle_file, f'{ass_name}', new_styles)
        remove_overlapping_subtitles(f'{ass_name}')

        try:
            create_video(image_file, audio_file, subtitle_file, output_file)
            print(f"Video created successfully: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    concept = input("What would you like to generate?")
    main(concept)