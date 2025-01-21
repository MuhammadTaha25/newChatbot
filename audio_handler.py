import torch
from transformers import pipeline
import librosa
import io

def convert_bytes_to_array(audio_bytes):
    """
    Converts audio bytes into a NumPy array for processing.
    
    Parameters:
        audio_bytes (bytes): The audio file in bytes format.
        
    Returns:
        numpy.ndarray: The audio signal as a NumPy array.
    """
    # Wrap the audio bytes in a BytesIO stream for librosa to read.
    audio_bytes = io.BytesIO(audio_bytes)
    
    # Load the audio data and extract the sample rate using librosa.
    audio, sample_rate = librosa.load(audio_bytes)
    print(sample_rate)  # Print the sample rate for debugging purposes.
    
    return audio  # Return the audio signal as a NumPy array.

def transcribe_audio(audio_bytes):
    """
    Transcribes audio bytes into text using a pre-trained Whisper model.
    
    Parameters:
        audio_bytes (bytes): The audio file in bytes format.
        
    Returns:
        str: The transcribed text from the audio.
    """
    # Select the appropriate device (GPU if available, otherwise CPU).
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize the Whisper ASR pipeline with the specified model and device.
    pipe = pipeline(
        task="automatic-speech-recognition",  # ASR task for transcribing speech.
        model="openai/whisper-small",        # Pre-trained Whisper model from OpenAI.
        chunk_length_s=30,                   # Process audio in 30-second chunks.
        device=device,                       # Device to perform computations on.
    )   

    # Convert the input audio bytes into a NumPy array.
    audio_array = convert_bytes_to_array(audio_bytes)
    
    # Perform the transcription using the pipeline and extract the text.
    prediction = pipe(audio_array, batch_size=1)["text"]
    
    return prediction  # Return the transcribed text.
