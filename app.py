import os
os.environ["NUMBA_DISABLE_CACHE"] = "1"

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from huggingface_hub import snapshot_download
import torch
import os
import uvicorn
import soundfile as sf
from vinorm import TTSnorm
from underthesea import sent_tokenize
from unidecode import unidecode
import string
from datetime import datetime
import requests

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = FastAPI()

# 📁 Load model on startup
print("🔄 Tải mô hình từ huggingface...")
snapshot_download(repo_id="epchannel/EpXTTS", repo_type="model", local_dir="model")

# ⬇️ Tải mẫu giọng nói nếu chưa có
def download_sample_voice():
    url = "https://huggingface.co/epchannel/EpXTTS/resolve/main/samples/bongxinh.wav"
    local_path = "bongxinh.wav"
    if not os.path.exists(local_path):
        print("⬇️  Tải file bongxinh.wav từ Hugging Face...")
        with open(local_path, "wb") as f:
            f.write(requests.get(url).content)
    else:
        print("✅ Đã có bongxinh.wav")

download_sample_voice()

# 🔁 Load model once
def load_model():
    config = XttsConfig()
    config.load_json("model/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path="model/model.pth", vocab_path="model/vocab.json", use_deepspeed=False)
    if torch.cuda.is_available():
        model.cuda()
    return model

def prepare_speaker_embedding(model):
    return model.get_conditioning_latents(
        audio_path="bongxinh.wav",
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )

model = load_model()
gpt_latent, speaker_embed = prepare_speaker_embedding(model)

# 🔡 Normalize
def normalize_vietnamese_text(text):
    return (
        TTSnorm(text, unknown=False, lower=False, rule=True)
        .replace("..", ".").replace("!.", "!").replace("?.", "?")
        .replace(" .", ".").replace(" ,", ",").replace('"', "")
        .replace("'", "").replace("AI", "Ây Ai").replace("A.I", "Ây Ai")
        .replace("anh/chị", "anh chị")
    )

def get_file_name(text, max_char=50):
    filename = unidecode(text[:max_char].lower().replace(" ", "_"))
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    return f"{timestamp}_{filename}.mp3"

# 🛠️ TTS Handler
def run_tts(text):
    text = normalize_vietnamese_text(text)
    sentences = sent_tokenize(text)
    wav_chunks = []

    for sent in sentences:
        if sent.strip() == "":
            continue
        wav_chunk = model.inference(
            text=sent,
            language="vi",
            gpt_cond_latent=gpt_latent,
            speaker_embedding=speaker_embed,
            temperature=0.5,
            top_k=20,
            top_p=0.85,
            repetition_penalty=5.0,
        )
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))

    final_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", get_file_name(text))
    sf.write(output_path, final_wav.squeeze(0).numpy(), 24000, format='MP3')
    return output_path

# 🚀 API endpoint
@app.post("/tts")
def tts_api(text: str = Form(...)):
    print(f"📥 Nhận yêu cầu TTS: {text[:60]}...")
    output_path = run_tts(text)
    return FileResponse(output_path, media_type="audio/mpeg", filename=os.path.basename(output_path))
