from pathlib import Path
from threading import Thread

import gdown
import gradio as gr
import librosa
import numpy as np
import torch

from pipeline import build_audiosep

CHECKPOINTS_DIR = Path("checkpoint")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The model will be loaded in the future
MODEL_NAME = CHECKPOINTS_DIR / "audiosep_base_4M_steps.ckpt"
MODEL = None


description = """
# AudioSep: Separate Anything You Describe
[[Project Page]](https://audio-agi.github.io/Separate-Anything-You-Describe) [[Paper]](https://audio-agi.github.io/Separate-Anything-You-Describe/AudioSep_arXiv.pdf) [[Code]](https://github.com/Audio-AGI/AudioSep)

AudioSep is a foundation model for open-domain sound separation with natural language queries.
AudioSep demonstrates strong separation performance and impressivezero-shot generalization ability on
numerous tasks such as audio event separation, musical instrument separation, and speech enhancement.
"""


def get_model():
    model = build_audiosep(
        config_yaml="config/audiosep_base.yaml",
        checkpoint_path=MODEL_NAME,
        device=DEVICE,
    )
    return model


def inference(audio_file_path: str, text: str):
    print(f"Separate audio from [{audio_file_path}] with textual query [{text}]")
    mixture, _ = librosa.load(audio_file_path, sr=32000, mono=True)

    with torch.no_grad():
        text = [text]

        conditions = MODEL.query_encoder.get_query_embed(
            modality="text", text=text, device=DEVICE
        )

        input_dict = {
            "mixture": torch.Tensor(mixture)[None, None, :].to(DEVICE),
            "condition": conditions,
        }

        sep_segment = MODEL.ss_model(input_dict)["waveform"]

        sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()

        return 32000, np.round(sep_segment * 32767).astype(np.int16)


def download_models():
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    success_file = CHECKPOINTS_DIR / "_SUCCESS"

    models = (
        (
            "https://drive.google.com/file/d/1wQuXThdATXrkmkPM2sRGaNapJ4mTqmlY/view?usp=sharing",
            MODEL_NAME,
        ),
        (
            "https://drive.google.com/file/d/11oj8_tPG6SXgw5fIEsZ5HiWZnJOrvdhw/view?usp=sharing",
            CHECKPOINTS_DIR / "music_speech_audioset_epoch_15_esc_89.98.pt",
        ),
    )

    def download(models):
        for model_url, model_path in models:
            gdown.download(model_url, str(model_path), quiet=False, fuzzy=True)

        success_file.touch()

        global MODEL
        MODEL = get_model()
        button.update(value="Separate", interactive=True)

    if not success_file.exists():
        thread = Thread(target=download, args=[models])
        thread.start()


with gr.Blocks(title="AudioSep") as demo:
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio()
            text = gr.Textbox()
        with gr.Column():
            with gr.Column():
                output_audio = gr.Audio(scale=10)
                button = gr.Button(
                    "Downloading the models...",
                    variant="primary",
                    scale=2,
                    size="lg",
                    interactive=False,
                )
                button.click(
                    fn=inference, inputs=[input_audio, text], outputs=[output_audio]
                )

download_models()

demo.queue().launch(share=True)
