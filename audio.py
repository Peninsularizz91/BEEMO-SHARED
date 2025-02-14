import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(base_dir, 'SavedIMG')

def list_audio_devices():
    devices = sd.query_devices()
    input_devices = [device for device in devices if device['max_input_channels'] > 0]
    for idx, device in enumerate(input_devices):
        print(f"{idx}: {device['name']}")

def select_audio_device(device_index):
    devices = sd.query_devices()
    input_devices = [device for device in devices if device['max_input_channels'] > 0]
    if 0 <= device_index < len(input_devices):
        selected_device = input_devices[device_index]
        sd.default.device = selected_device['name']
        print(f"Selected device: {selected_device['name']}")
        return selected_device['max_input_channels'], selected_device
    else:
        print("Invalid device index")
        return None, None

def record_audio(filename, duration=10, channels=1, api=None):
    audio_path = os.path.join(save_folder, filename)
    try:
        fs = 44100  # Sample rate
        print(f"Recording for {duration} seconds with {channels} channel(s)...")
        sd.default.channels = channels
        if api:
            sd.default.device = (None, api)
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float64')
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")
        write(filename, fs, audio_data)
    except Exception as e:
        print(f"Error during recording: {e}")

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os

def audio_to_spectrogram(audio_path, spectrogram_path):
    try:
        sample_rate, samples = wavfile.read(audio_path)
        plt.figure(figsize=(10, 6))
        plt.specgram(samples, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='magma')
        plt.axis('off')
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return spectrogram_path
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None

def get_next_filename(base_path):
    files = [f for f in os.listdir(base_path) if f.startswith("BeemoDosSpectrogram_") and f.endswith(".png")]
    numbers = [int(f.split('_')[-1].split('.')[0]) for f in files]
    next_number = max(numbers) + 1 if numbers else 1
    return os.path.join(base_path, f"BeemoDosSpectrogram_{next_number}.png")

def rename_file_based_on_content(spectrogram_path, base_path):
    try:
        new_path = get_next_filename(base_path)
        os.rename(spectrogram_path, new_path)
        print(f"File renamed to {new_path}")
        return new_path
    except Exception as e:
        print(f"Error renaming file: {e}")
        return spectrogram_path


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile

# def audio_to_spectrogram(audio_path, spectrogram_path):
#     try:
#         sample_rate, samples = wavfile.read(audio_path)
#         print(f"Sample Rate: {sample_rate}, Samples Shape: {samples.shape}")

#         # If samples have multiple channels, average them into a single channel
#         if len(samples.shape) > 1:
#             samples = np.mean(samples, axis=1)
        
#         plt.figure(figsize=(10, 6))
#         plt.specgram(samples, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='magma')
#         plt.axis('off')
#         plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
#         plt.close()

#         return spectrogram_path
#     except Exception as e:
#         print(f"Error generating spectrogram: {e}")
#         return None

# def get_next_filename(base_path):
#     files = [f for f in os.listdir(base_path) if f.startswith("BeemoDosSpectrogram_") and f.endswith(".png")]
#     numbers = [int(f.split('_')[-1].split('.')[0]) for f in files]
#     next_number = max(numbers) + 1 if numbers else 1
#     return os.path.join(base_path, f"BeemoDosSpectrogram_{next_number}.png")

# def rename_file_based_on_content(spectrogram_path, base_path):
#     try:
#         new_path = get_next_filename(base_path)
#         os.rename(spectrogram_path, new_path)
#         print(f"File renamed to {new_path}")
#         return new_path
#     except Exception as e:
#         print(f"Error renaming file: {e}")
#         return spectrogram_path
