
import os
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
from groq import Groq
from transformers import BarkModel,AutoProcessor
import torch
from dotenv import load_dotenv, find_dotenv
load_dotenv(".env")
api_key  = os.getenv("APIKEY")
client = Groq(
    api_key=api_key,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
voice_preset = "v2/en_speaker_6"
concept = "depth first search"
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"Teach me how to write {concept} in python. Put the code snippet specifically in brackets. Please only provide 1 code snippet. Please fill in this template pretending you are explaining this code snippet to someone (DO NOT MODIFY ANYWHERE ELSE IN THE TEMPLATE EXCEPT SQUARE BRACKETS):\
             Here is how to write [INSERT CONCEPT (DO NOT PUT SHORT FORM FULL NAME ONLY)]:\
             ``` [PUT FULL CODE SNIPPET HERE, NOWHERE ELSE, DO NOT PUT CODE BELOW]``` \
            [00:00-00:05] [INSERT SCRIPT THAT EXPLAINS THE FIRST 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS BREIF AS POSSIBLE]\
            [00:05-00:10] [INSERT SCRIPT THAT EXPLAINS THE NEXT 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS BREIF AS POSSIBLE]\
            [00:10-00:15] [INSERT SCRIPT THAT EXPLAINS THE NEXT 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS BREIF AS POSSIBLE]\
            [00:15-00:20] [INSERT SCRIPT THAT EXPLAINS THE NEXT 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS BREIF AS POSSIBLE]\
            [00:20-00:25] [INSERT SCRIPT THAT EXPLAINS THE NEXT 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS BREIF AS POSSIBLE]\
            [00:25-00:30] [INSERT SCRIPT THAT EXPLAINS THE LAST 10% OF THE CODE YOU WROTE, DO NOT PUT CODE HERE ONLY EXPLANATION, KEEP THIS BREIF AS POSSIBLE]\
            ",
        }
    ],
    model="llama3-8b-8192",
)

content = chat_completion.choices[0].message.content
print(content)
content = content.split("```")
intro , code, scripts= content[0], content[1],content[-1]
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
    inputs = processor(l, voice_preset=voice_preset)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy()
    full_audio = np.hstack((full_audio,audio_array))

# import ipdb; ipdb.set_trace()
write("example.wav",sampling_rate,  data = (full_audio.T*32767).astype("int16"))

# sf.write('output_audio.wav', full_audio, sampling_rate)





