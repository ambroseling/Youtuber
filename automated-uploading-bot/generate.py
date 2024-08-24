import traceback
import os
import io
from PIL import Image
import numpy as np
import srt
import re
import ffmpeg
import whisperx
import requests
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
import random
from main import start_typing
import time


# print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))

def get_audio_duration_ffmpeg(file_path):    
    '''
    Get the duration of an audio file using FFmpeg's ffprobe tool.

    ## Parameters:
    file_path (str): The path to the audio file.

    ## Returns: 
    float: The duration of the audio file in seconds.

    This function uses `ffprobe`, a multimedia stream analyzer from the FFmpeg project,
    to obtain the duration of an audio file. It runs the `ffprobe` command with appropriate
    arguments to fetch the duration information, processes the output, and returns the
    duration as a float value representing the number of seconds.

    ## Example:
    --------
    >>> duration = get_audio_duration_ffmpeg("sample.mp3")
    >>> print(duration)
    180.5 '''
    
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
        # Decode the output and split by lines
    output = result.stdout.decode("utf-8").strip().split("\n")
    
    # The duration is typically the last line, so return it as a float
    duration = float(output[-1])
    
    return duration
    return float(result.stdout)


def speed_up_audio(input_file, output_file, speed_factor):
    """
    Speeds up the audio of a given input file by a specified factor and saves the modified audio to an output file.

    ## Parameters:
    - input_file (str): The path to the input audio file.
    - output_file (str): The path where the sped-up audio file will be saved.
    - speed_factor (float): The factor by which to speed up the audio. For example, a speed_factor of 1.5 will increase
      the playback speed by 50%, while a speed_factor of 0.5 will slow it down by 50%.

    ## Returns:
    None

    ## Example:
    speed_up_audio('input.wav', 'output.wav', 1.5)
    This will take 'input.wav', speed up the audio by 50%, and save the result as 'output.wav'.
    """
    # Construct the ffmpeg command
    command = [
        'ffmpeg',
        '-i', input_file,       # Input file
        '-filter:a', f"atempo={speed_factor}",  # Speed up audio
        '-vn',                  # No video
        '-y',                   # Overwrite output file
        output_file
    ]
    
    # Run the command
    subprocess.run(command)


def create_video(image_file, audio_file, subtitle_file, output_file):
    """
    ## Description:
    Creates a 40-second video by combining an image, audio, and subtitle file using FFmpeg.

    ## Parameters:
    - `image_file` (str): The path to the input image file that will be used as the video background.
    - `audio_file` (str): The path to the input audio file that will play in the background of the video.
    - `subtitle_file` (str): The path to the ASS subtitle file to be overlaid on the video.
    - `output_file` (str): The path where the output video file will be saved.

    ## Functionality:
    - The function uses FFmpeg to loop the image as the video background.
    - Combines it with the audio.
    - Overlays subtitles.
    - Saves the result as a 40-second MP4 video.

    ## Returns:
    - `None`

    ## Example:
    `create_video('image.png', 'audio.mp3', 'subtitles.ass', 'output.mp4')`
    - This will combine 'image.png' as a looping background, 'audio.mp3' as the audio track, 
    and 'subtitles.ass' as the subtitle file into a 40-second video saved as 'output.mp4'.
    """
    command = [
        'ffmpeg',
        '-loop', '1',                   # Loop the image
        '-i', image_file,               # Input image
        '-i', audio_file,               # Input audio
        '-vf', f"ass={subtitle_file}",  # Video filter for ASS subtitles
        '-c:v', 'libx264',              # Video codec
        '-c:a', 'aac',                  # Audio codec
        '-pix_fmt', 'yuv420p',           # Pixel format
        '-b:a', '192k',                 # Audio bitrate
        '-movflags', 'faststart',       # Optimize for streaming
        '-t', '40',                     # Set duration to 40 seconds
        '-f', 'mp4',                    # Output format
        output_file                     # Output file
    ]

    # Run the command
    subprocess.run(command, check=True)


def update_ass_styles(input_file, output_file, new_styles):
    """
    ## Description:
    Updates the styles in an ASS subtitle file by modifying the `[V4+ Styles]` section based on the provided new styles.

    ## Parameters:
    - `input_file` (str): The path to the input ASS subtitle file.
    - `output_file` (str): The path where the updated ASS subtitle file will be saved.
    - `new_styles` (dict): A dictionary containing the new styles to be applied. 
    - The keys are style names, and the values are dictionaries with style attributes as keys and their new values as values.

    ## Functionality:
    - Reads the input ASS file and identifies the `[V4+ Styles]` section.
    - Searches for style lines and updates them according to the `new_styles` dictionary.
    - Writes the modified content to the output file.

    ## Returns:
    - `None`

    ## Example:
    ```python
    new_styles = {
        'Default': {1: 'Arial', 2: '24'},  # Change font to Arial and font size to 24 for 'Default' style
        'Subtitle': {1: 'Times New Roman', 2: '20'}  # Change font to Times New Roman and font size to 20 for 'Subtitle' style
    }
    update_ass_styles('input.ass', 'output.ass', new_styles)
    ```
    - This will read 'input.ass', update the styles specified in the `new_styles` dictionary, and save the changes to 'output.ass'.
    """

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
    """
    ## Description:
    Removes overlapping subtitles from an ASS subtitle file by checking for duplicate start times and eliminating entries with the same start time.

    ## Parameters:
    - `file_path` (str): The path to the ASS subtitle file to be processed.

    ## Functionality:
    - Reads the ASS subtitle file line by line.
    - Identifies lines that contain subtitles using a regex pattern (`Dialogue:.*`).
    - Checks for duplicate start times among the subtitles and removes any overlapping entries.
    - Writes the cleaned list of subtitles back to the same file.

    ## Returns:
    - `None`

    ## Example:
    `remove_overlapping_subtitles('subtitles.ass')`
    - This will process 'subtitles.ass' and remove any overlapping subtitles based on start times, writing the cleaned content back to 'subtitles.ass'.
    """
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


def generate_assets(hf_token, payload):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    while True:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 503:
            error_data = response.json()
            estimated_time = error_data.get('estimated_time', 60)
            print(f"Model is still loading. Waiting for {estimated_time} seconds...")
            time.sleep(estimated_time)
        else:
            response.raise_for_status()  # Raise an exception if the response was not successful
            return response.content


def transcribe_audio(audio_file_new,subtitle_file,whisper_model, style = 'Default'):
    """
    ## Description:
    Transcribes an audio file into subtitles with word-by-word highlighting, aligns the transcription using WhisperX, and updates the subtitles file with custom styles. This function also removes overlapping subtitles.

    ## Parameters:
    - `audio_file_new` (str): The path to the new audio file to be transcribed.
    - `subtitle_file` (str): The path where the generated subtitle file will be saved and updated.
    - `whisper_model` (Whisper Model Object): The Whisper model used for transcribing the audio.
    - `style` (str): The name of the style to be applied to the subtitles. Default is 'Default'.

    ## Styles Available:
    - **Default**: Basic style with green text, thin outline, and slight shadow.
    - **Bold**: Uses Arial Black font with red primary color, thicker outline, bold text.
    - **Italic**: Uses Georgia font with green text, italicized style.
    - **Highlight**: Yellow text with a thick outline for highlighting purposes.
    - **Shadowed**: White text with a significant shadow effect for depth.
    - **OutlineThick**: Black text with a thick red outline for a bold appearance.
    - **Glow**: White text with a yellow glow and shadow, giving a soft illuminated look.
    - **Mono**: Monospaced Courier New font for a typewriter or coding terminal effect.
    - **Comic**: Comic Sans MS font with orange text, providing a fun, casual style.
    - **Fancy**: Times New Roman font with pink text, adding an elegant and formal look.
    - **Retro**: Lucida Console font with gold text and red background, giving a retro feel.
    - **Minimalist**: Helvetica font with white text and no outline or shadow for a clean, simple style.

    ## Functionality:
    - Loads and transcribes the audio file using the specified Whisper model.
    - Aligns the transcribed text segments with the audio using WhisperX.
    - Generates subtitles in `.srt` format and saves them to the specified location.
    - Applies custom styles to the subtitle file and highlights words as they are spoken.
    - Removes overlapping subtitles to ensure a clean output.

    ## Returns:
    - `None`

    ## Example:
    ```python
    whisper_model = load_whisper_model()  # Example function to load your Whisper model
    transcribe_audio('audio_new.wav', 'subtitles.ass', whisper_model, style='Default')
    ```
    - This will transcribe 'audio_new.wav' to generate subtitles, align them, apply the 'Default' style, and remove overlaps in 'subtitles.ass'.
    """
    
    batch_size = 16
    device = "cuda"
    if os.path.exists(audio_file_new):
        audio = whisperx.load_audio(audio_file_new)
        result = whisper_model.transcribe(audio, batch_size=batch_size)
        language_code=result["language"]
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        result['language'] = language_code
        vtt_writer = get_writer("srt", f'{concept.replace(" ","_")}/')
        vtt_writer(
            result,
            audio_file_new,
            {"max_line_width": 15, "max_line_count": 1, "highlight_words": True},
        )
        if 'intro' in subtitle_file:
            file_name = f'{concept.replace(" ","")}intro_new.srt'
        else:
            file_name = f'{concept.replace(" ","")}new.srt'
        (ffmpeg.input(os.path.join(concept.replace(" ",""),file_name)).output(subtitle_file).run())

        # Usage
        # ass_name = os.path.join(concept.replace(" ","_"),f"{concept.replace(" ","_")}.ass")
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
    },
    "Bold": {
        1: "Arial Black",       # Fontname
        2: "14",                # Fontsize
        3: "&H0000FFFF",        # PrimaryColour (red)
        4: "&H00FFFFFF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H64000000",        # BackColour
        16: "3",                # Outline (thickness)
        17: "2",                # Shadow (depth)
        21: "1"                 # Bold
    },
    "Italic": {
        1: "Georgia",           # Fontname
        2: "12",                # Fontsize
        3: "&H00FF00FF",        # PrimaryColour (green)
        4: "&H000000FF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H64000000",        # BackColour
        16: "1",                # Outline (thickness)
        17: "0",                # Shadow (depth)
        21: "1"                 # Italic
    },
    "Highlight": {
        1: "Arial",             # Fontname
        2: "12",                # Fontsize
        3: "&H00FFFF00",        # PrimaryColour (yellow)
        4: "&H000000FF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H64000000",        # BackColour
        16: "4",                # Outline (thickness)
        17: "0"                 # Shadow (depth)
    },
    "Shadowed": {
        1: "Verdana",           # Fontname
        2: "12",                # Fontsize
        3: "&H00FFFFFF",        # PrimaryColour (white)
        4: "&H000000FF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H64000000",        # BackColour
        16: "1",                # Outline (thickness)
        17: "4"                 # Shadow (depth)
    },
    "OutlineThick": {
        1: "Tahoma",            # Fontname
        2: "14",                # Fontsize
        3: "&H00000000",        # PrimaryColour (black)
        4: "&H00FFFFFF",        # SecondaryColour (white)
        5: "&H00FF0000",        # OutlineColour (red)
        6: "&H64000000",        # BackColour
        16: "6",                # Outline (thickness)
        17: "0"                 # Shadow (depth)
    },
    "Glow": {
        1: "Calibri",           # Fontname
        2: "13",                # Fontsize
        3: "&H00FFFFFF",        # PrimaryColour (white)
        4: "&H0000FFFF",        # SecondaryColour (blue)
        5: "&H00000000",        # OutlineColour (black)
        6: "&H64FFFF00",        # BackColour (yellow glow)
        16: "2",                # Outline (thickness)
        17: "3"                 # Shadow (depth)
    },
    "Mono": {
        1: "Courier New",       # Fontname
        2: "11",                # Fontsize
        3: "&H00FFFFFF",        # PrimaryColour (white)
        4: "&H000000FF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H64000000",        # BackColour
        16: "0",                # Outline (thickness)
        17: "0"                 # Shadow (depth)
    },
    "Comic": {
        1: "Comic Sans MS",     # Fontname
        2: "16",                # Fontsize
        3: "&H00FFA500",        # PrimaryColour (orange)
        4: "&H000000FF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H64000000",        # BackColour
        16: "2",                # Outline (thickness)
        17: "1"                 # Shadow (depth)
    },
    "Fancy": {
        1: "Times New Roman",   # Fontname
        2: "18",                # Fontsize
        3: "&H00FFC0CB",        # PrimaryColour (pink)
        4: "&H000000FF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H64FFFFFF",        # BackColour (white)
        16: "1",                # Outline (thickness)
        17: "2"                 # Shadow (depth)
    },
    "Retro": {
        1: "Lucida Console",    # Fontname
        2: "15",                # Fontsize
        3: "&H00FFD700",        # PrimaryColour (gold)
        4: "&H000000FF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H64FF0000",        # BackColour (red)
        16: "2",                # Outline (thickness)
        17: "1"                 # Shadow (depth)
    },
    "Minimalist": {
        1: "Helvetica",         # Fontname
        2: "10",                # Fontsize
        3: "&H00FFFFFF",        # PrimaryColour (white)
        4: "&H000000FF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H00000000",        # BackColour
        16: "0",                # Outline (thickness)
        17: "0"                 # Shadow (depth)
    }
}
        update_ass_styles(subtitle_file, subtitle_file, new_styles)
        remove_overlapping_subtitles(subtitle_file)


def make_file_path(concept, extension):
    """
    ## Description:
    Generates a file path for a given concept and file extension, ensuring the directory exists.

    ## Parameters:
    - `concept` (str): The concept name used to create a directory and generate the file path.
    - `extension` (str): The file extension (e.g., 'wav', 'ass', 'mp4') for which the path is needed.

    ## Returns:
    - `str`: The generated file path.
    
    ## Example:
    ```python
    audio_file = file_path('Example Concept', 'wav')
    subtitle_file = file_path('Example Concept', 'ass')
    output_file = file_path('Example Concept', 'mp4')
    ```
    """

    # Rplace spaces with underscores in the concept name
    concept_dir = concept.replace(" ", "_")
    
    # Creat the directry if it doesnt exist
    if not os.path.isdir(concept_dir):
        os.mkdir(concept_dir)
    
    # Generate the file pth with the specified extension
    return os.path.join(concept_dir, f'{concept_dir}.{extension}')


def get_b_rolls(concept, num_b_rolls, hf_token):
    """
    ## Description:
    Generates a specified number of B-roll images based on a concept and saves them in a directory named after the concept.

    ## Parameters:
    - `concept` (str): The concept or theme to be included in the B-roll images.
    - `num_b_rolls` (int): The number of B-roll images to generate.
    - `hf_token` (str): The Hugging Face API token used to generate images.

    ## Returns:
    - `list`: A list of paths to the generated B-roll images.

    ## Example:
    ```python
    hf_token = "your_huggingface_api_token"
    concept = "Machine Learning"
    num_b_rolls = 3
    b_roll_paths = get_b_rolls(concept, num_b_rolls, hf_token)
    ```
    """

    funny_ppl = [
        "Barney the purple dinosaur from 'Barney and Friends childrens show'",
        "Donald Trump",
        "Snoop Dogg",
        "Mickey Mouse",
        "Goofy",
        "Courage the Cowardly Dog",
        "Mr. Bean",
        "Teletubbies",
        "LEGO Batman",
        "Kung Fu Panda",
        "Dumbo the elephant with big ears and a cute face",
        "Winnie the Pooh eating honey", 
        "Buzz lightyear from Disney's Toy Story eating pancakes on a table",
        "Woody from Disney's Toy Story"
    ]

    b_rolls = []
    num_images = 0
    concept_dir = concept.replace(" ", "_")

    # Ensure the concept directory exists
    if not os.path.isdir(concept_dir):
        os.mkdir(concept_dir)

    while num_images < num_b_rolls:
        try:
            # Randomly select a person from the funny_ppl list
            person = random.choice(funny_ppl)
            prompt = {"inputs": f"{person} on a computer coding with the screen saying: {concept}"}
            
            # Generate the image asset
            content = generate_assets(hf_token=hf_token, payload=prompt)
            image = Image.open(io.BytesIO(content))

            # Save the generated image to the concept directory
            image_path = os.path.join(concept_dir, f"b_roll_{num_images}.png")
            image.save(image_path)
            b_rolls.append(image_path)
            num_images += 1

        except Exception as e:
            print("Something went wrong with generating your B-roll images :(")
            print(f"Error: {str(e)}")
            traceback.print_exc()

    return b_rolls


def generate_explanation_and_summary(client, concept, model="llama3-8b-8192", voice_preset="v2/en_speaker_6"):
    """
    ## Description:
    Generates two chat completions using a language model:
    1. A detailed explanation with a code snippet on how to write the given concept in Python.
    2. A 30-word summary of the concept in full sentences.

    ## Parameters:
    - `client`: The client object used to communicate with the language model API.
    - `concept` (str): The concept to be explained and summarized.
    - `model` (str): The model name to be used for generating completions. Default is "llama3-8b-8192".
    - `voice_preset` (str): The voice preset to be used. Default is "v2/en_speaker_6".

    ## Returns:
    - `dict`: A dictionary containing the detailed explanation and the 30-word summary.
    
    ## Example:
    ```python
    client = your_client_object
    concept = "loops"
    response = generate_explanation_and_summary(client, concept)
    detailed_explanation = response["detailed_explanation"]
    summary = response["summary"]
    ```
    """

    # Generate detailed explanation with code snippet
    try: 
        detailed_explanation = client.chat.completions.create(
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
                    [00:25-00:30] [INSERT SCRIPT THAT EXPLAINS THE LAST 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS EXTREMELY SHORT, LESS THAN 15 WORDS]",
                },
            ],
            model=model,
        )
    except client.APIConnectionError as e:
        print("The server could not be reached")
        print(e.cause)  # an underlying Exception, likely raised within httpx.
    except client.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except client.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)

    # Generate 30-word summary
    summary = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Summarize this concept in 30 words in full sentences: {concept}",
            }
        ],
        model=model,
    )

    return {
        "detailed_explanation": detailed_explanation,
        "summary": summary
    }





def save_code_snippet_to_file(content, concept):
    """
    ## Description:
    Extracts a code snippet from the provided content, processes it to remove any line containing the word "python",
    and saves the snippet to a text file in a directory named after the concept.

    ## Parameters:
    - `content` (str): The content from which the code snippet is extracted. This content typically comes from an LLM output.
    - `concept` (str): The concept or theme used to name the directory and the file where the code snippet is saved.

    ## Returns:
    - `str`: The path to the saved code snippet file.
    
    ## Example:
    ```python
    content = "some LLM output with a code snippet in triple backticks"
    concept = "loops"
    code_file_path = save_code_snippet_to_file(content, concept)
    print(f"Code snippet saved to {code_file_path}")
    ```
    """

    # Extracting the code snippet from the LLM output
    try:
        code_snippet = content.split("```")[1].strip()
    except IndexError:
        raise ValueError("No code snippet found in the provided content.")

    code_lines = code_snippet.splitlines()

    # Check if any line contains the word "python"
    if any("python" in line.lower() for line in code_lines):
        # Remove the first line if "python" is found
        code_lines = "\n".join(code_lines[1:])
    else:
        code_lines = "\n".join(code_lines[:])

    # Prepare the directory for the concept
    concept_dir = concept.replace(" ", "_")
    if not os.path.isdir(concept_dir):
        os.mkdir(concept_dir)

    # Save the code snippet to a text file
    code_file_path = os.path.join(concept_dir, f"{concept}_code.txt")
    with open(code_file_path, 'w') as code_file:
        code_file.write(code_lines)

    print(f"Code snippet saved to {code_file_path}")

    return code_file_path

# TEMPORARY:
def load_code_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            code = file.read()
            return code
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None



def main(concept):

    # making the files we need:
    audio_file = make_file_path(concept, 'wav')          # Path to the audio file
    subtitle_file = make_file_path(concept, 'ass')       # Path to the ASS subtitle file
    output_file = make_file_path(concept, 'mp4')         # Path to the output video file
    
    # getting the API keys and setting up the inference models not stored locally:
    load_dotenv(".env")
    Groq_api_key  = os.getenv("GROQ_APIKEY")
    import ipdb; ipdb.set_trace()
    hf_token = os.getenv("HF_TOKEN")
    client = Groq(api_key = Groq_api_key,)

    # # get the B-Rolls:
    # num_b_rolls = 3
    # b_roll_paths = get_b_rolls(concept, num_b_rolls, hf_token)

    # setting person/speaker's voice
    voice_preset = "v2/en_speaker_6"
    
    # getting the content and summary: 
    response = generate_explanation_and_summary(client, concept)

    detailed_explanation_response = response["detailed_explanation"]
    summary_response = response["summary"]
    
    content = detailed_explanation_response.choices[0].message.content
    summary =  summary_response.choices[0].message.content
    summary = summary.split("\n")[-1]
    print(content)
    print(f"Summary: {summary}")


    # loading up the models:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark-small").to(device)
    whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)


    # # gets the code into a text file:
    # code_file_path = save_code_snippet_to_file(concept, content)
    # code_snippet = load_code_from_file(code_file_path)

    # run this method here, imported from main.py, which I need to be able to run in command prompt AKA windows. Just that one bit of code.
    # print("trying typing now:")
    # start_typing(code_snippet) 


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

    summary_audio = None
    # generate audio for the summary
    print("Generating audio for the intro....")
    inputs = processor(summary, voice_preset=voice_preset).to(device)
    intro_audio = model.generate(**inputs)
    intro_audio = intro_audio.cpu().numpy()
    intro_audio_file = audio_file.replace("/"+concept.replace(" ","_"),"/"+concept.replace(" ","_")+"_intro")
    intro_audio_file_new = intro_audio_file.replace("intro","intro_new")
    intro_subtitle_file = subtitle_file.replace("/"+concept.replace(" ","_"),"/"+concept.replace(" ","_")+"_intro")
    intro_output_file = output_file.replace("/"+concept.replace(" ","_"),"/"+concept.replace(" ","_")+"_intro")
    write(intro_audio_file,sampling_rate,  data = (intro_audio.T*32767).astype("int16"))
    duration = get_audio_duration_ffmpeg(intro_audio_file)
    # import ipdb; ipdb.set_trace()
    duration_target = 10
    speed_factor = duration / duration_target
    speed_up_audio(intro_audio_file,intro_audio_file_new,speed_factor=speed_factor)
    transcribe_audio(audio_file_new=intro_audio_file_new,subtitle_file=intro_subtitle_file,whisper_model=whisper_model)
    try:
        create_video(f'{concept.replace(" ","_")}/b_roll_0.png' ,intro_audio_file_new, intro_subtitle_file, intro_output_file)
        print(f"Later half of video created successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


    # generate audio for the 5 lines in the script
    print("Generating audio for the script....")
    for line in script_wtimes:
        l = script_wtimes[line]
        text += l
        inputs = processor(l, voice_preset=voice_preset).to(device)

        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy()
        full_audio = np.hstack((full_audio,audio_array))

    # import ipdb; ipdb.set_trace()
    write(audio_file,sampling_rate,  data = (full_audio.T*32767).astype("int16"))
    duration = get_audio_duration_ffmpeg(audio_file)
    # import ipdb; ipdb.set_trace()
    duration_target = 40
    speed_factor = duration / duration_target
    audio_file_new = os.path.join(f'{concept.replace(" ","_")}',f'{concept.replace(" ","_")}_new.wav')
    speed_up_audio(audio_file,audio_file_new,speed_factor=speed_factor)
    transcribe_audio(audio_file_new=audio_file_new,subtitle_file=subtitle_file,whisper_model=whisper_model)

    try:
        create_video(f'{concept.replace(" ","_")}/b_roll_1.png' ,audio_file_new, subtitle_file, output_file)
        print(f"Later half of video created successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    concept = input("What would you like to generate?:  ")
    main(concept)