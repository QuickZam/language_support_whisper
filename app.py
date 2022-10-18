import torch
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = whisper.load_model("large")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model 
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    la = model_inputs.get("language") 
    if mp3BytesString == None:
        return {'message': "No input provided"}
    
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    translate_options = dict(task="translate", language = la)
    result = model.transcribe("input.mp3", **translate_options)
    return result 
