import gradio as gr
from transformers import pipeline
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import tempfile

# -------- LYRICS AI --------
lyrics_ai = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct"
)

def lyrics_ai_fn(prompt):
    text = f"Write 100% copyright free Hindi song lyrics: {prompt}"
    out = lyrics_ai(text, max_length=300)
    return out[0]["generated_text"]

# -------- MUSIC GENERATE AI --------
music_model = MusicGen.get_pretrained("musicgen-small")
music_model.set_generation_params(duration=30)

def music_generate_fn(prompt):
    music = music_model.generate([prompt])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_write(tmp.name.replace(".wav",""), music[0], music_model.sample_rate)
    return tmp.name

# -------- MUSIC CONVERT AI --------
def music_convert_fn(audio, style):
    prompt = f"{style} style instrumental music"
    music = music_model.generate([prompt])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_write(tmp.name.replace(".wav",""), music[0], music_model.sample_rate)
    return tmp.name

# -------- UI --------
with gr.Blocks() as demo:
    gr.Markdown("# 🎵 AI Music & Lyrics Tool")

    option = gr.Radio(
        ["Lyrics AI", "Music Generate AI", "Music Convert AI"],
        label="Select Option"
    )

    prompt = gr.Textbox(label="Prompt / Style")
    audio = gr.Audio(type="filepath", label="Upload Music (Convert only)")

    out_text = gr.Textbox(label="Text Output")
    out_audio = gr.Audio(label="Audio Output")

    def controller(option, prompt, audio):
        if option == "Lyrics AI":
            return lyrics_ai_fn(prompt), None
        if option == "Music Generate AI":
            return "Music generated 👇", music_generate_fn(prompt)
        if option == "Music Convert AI":
            return "Converted music 👇", music_convert_fn(audio, prompt)

    btn = gr.Button("Generate / Convert")
    btn.click(controller, [option, prompt, audio], [out_text, out_audio])

demo.launch()
