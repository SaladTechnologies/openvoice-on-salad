from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import Optional
from azure.storage.blob import ContainerClient
import torch
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter
import os
import datetime


def azure_initiate(
    result_blob: str,
    storage_connection_string: str,
):
    azure_client = ContainerClient.from_connection_string(
        storage_connection_string, result_blob
    )
    return azure_client


# Check for CUDA device and set it
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model: 
ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
output_dir = '.data/outputs'
voice_dir = '.data/input_voice'
text_dir = '.data/input_text'
tts_dir = '.data/outputs/tts'
clone_dir = '.data/outputs/clone'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(voice_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)
os.makedirs(tts_dir, exist_ok=True)
os.makedirs(clone_dir, exist_ok=True)
os.makedirs('.data/tmp', exist_ok=True)
# load the base speaker tts model
base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')
source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
app = FastAPI()


def inference(connection_string: str, input_container_name: str, output_container_name: str, voices_container_name: Optional[str] = None, reference_voice: Optional[str] = None, speed: Optional[float] = 1.0, language: Optional[str] = "English", speaker_tone: Optional[str] = "default", text_file: Optional[str] = None):
    # authentiacate in azure
    clone = False
    if voices_container_name and reference_voice:
        clone = True
    if clone is True:
        voices_blob = azure_initiate(voices_container_name, connection_string)
    if input_container_name:
        input_blob = azure_initiate(input_container_name, connection_string)
    result_blob = azure_initiate(output_container_name, connection_string)

    process_start_time = datetime.datetime.now()

    # start processing

    # if input text was not yet downloaded download:
    if text_file not in os.listdir(text_dir):
        # download blob with name text_file from input_container_name
        blob_client = input_blob.get_blob_client(text_file)
        # blob_client = input_blob.get_blob_client(text_file).download_blob()
        data = blob_client.download_blob().readall()
        with open(f"{text_dir}/{text_file}", "wb") as f:
            f.write(data)
    # read the text file
    with open(f"{text_dir}/{text_file}", "r") as f:
        text = f.read()
    tts_file_name = text_file.rsplit('.', 1)[0] + ".wav"
    tts_path = f"{tts_dir}/{tts_file_name}"
    # Step 1: TTS with base speaker
    base_speaker_tts.tts(text, tts_path, speaker=speaker_tone, language=language, speed=speed)
    result_path = tts_path
    result_file_name = tts_file_name

    # Step 2: Voice Cloning
    if clone is True:
        voice_file_path = f'{voice_dir}/{reference_voice}'
        # download the voice file
        if reference_voice not in os.listdir(voice_dir):
            blob_client = voices_blob.get_blob_client(reference_voice)
            voice_file = blob_client.download_blob()
            # save voice file to voice_file_path
            with open(voice_file_path, "wb") as my_blob:
                voice_file.readinto(my_blob)
        target_se, audio_name = se_extractor.get_se(voice_file_path, tone_color_converter, target_dir='.data/tmp', vad=True)
        encode_message = "@MyShell"
        clone_result_name = f"{text_file.rsplit('.', 1)[0]}_cloned.wav"
        clone_result_path = f"{clone_dir}/{clone_result_name}"
        tone_color_converter.convert(
            audio_src_path=tts_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=clone_result_path,
            message=encode_message)
        result_path = clone_result_path
        result_file_name = clone_result_name
    
    # save result to azure
    end_time = datetime.datetime.now()
    output_blob_client = result_blob.get_blob_client(result_file_name)
    # upload wav file from local path to blob
    with open(result_path, "rb") as bytes_data:
        output_blob_client.upload_blob(bytes_data, overwrite=True)

    return {"status": "success", "result saved to": f"{output_container_name}/{result_file_name}", "processing time": str(end_time - process_start_time)}

@app.get("/hc")
async def health_check():
    return {"status": "healthy"}

@app.post("/process")
async def process(
    connection_string: str = Query("DefaultEndpointsProtocol=https;AccountName=accountname;AccountKey=key;EndpointSuffix=core.windows.net", description="Azure Storage Connection String"),
    input_container_name: str = Query("requests", description="Container name for input files"),
    output_container_name: str = Query("results", description="Container name for output files"),
    voices_container_name: Optional[str] = Query("voices", description="Container name for voice files"),
    reference_voice: Optional[str] = Query(None, description="Voice file to be used as reference"),
    speed: float = Query(1.0, description="Speed of the voice"),
    language: str = Query("English", description="Language of the voice"),
    speaker_tone: str = Query("default", description="Tone of voice. Options: default, whispering, shouting, excited, cheerful, terrified, angry, sad, friendly"),
    text_file: str = Query(description="Text file to be used for TTS"),
    # processing_container_name: str = Query("preprocess", description="Container name for processing files"),
):
    inference(connection_string, input_container_name, output_container_name, voices_container_name, reference_voice, speed, language, speaker_tone, text_file)
    # Here you can use the parameters to do whatever processing you need
    # For now, it just returns the parameters as they were received
    return {"status": "success"}