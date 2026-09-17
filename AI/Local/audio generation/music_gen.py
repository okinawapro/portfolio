import os
import torch
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

def generate_local_music(
    prompt="epic cinematic soundtrack uplifting and motivational",
    #1 min 1536 tokens, 30seg 768 tokens
    #higher than 2000 would crash the program
    max_new_tokens=636,
    output_file="local_music.wav"
):
    # Force offline mode (prevents ANY HF Hub calls)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_path = "./musicgen-small"  # local folder with model files

    # Load processor and model ONLY from local directory
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = MusicgenForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True
    )

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare input
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate audio tokens locally
    audio_values = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )

    # Extract audio
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio = audio_values[0, 0].cpu().numpy()

    # Save WAV
    wavfile.write(output_file, rate=sampling_rate, data=audio)

    print(f"Local music generated and saved to {output_file}")
    return audio, sampling_rate


if __name__ == "__main__":
    generate_local_music()
