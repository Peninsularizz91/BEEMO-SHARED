import os
import numpy as np
import time
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import discord
from discord.ext import commands
import asyncio
import logging
from google.oauth2 import service_account
import sounddevice as sd

# Parameters
CHANNELS = 1              # Number of audio channels (mono)
RATE = 44100             # Sampling rate (samples per second)
CHUNK = 1024             # Number of frames per buffer
DURATION = 15            # Recording duration in seconds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("frequency")

# Google Sheets parameters
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1h387i_m0wb2RQ8zrO-gEwHPGzs0mp8qtgYjxXO7Gg00"
RANGE_NAME = "Sheet1!A:G"

# Function to authenticate and connect to Google Sheets
def connect_to_google_sheets():
    # Update the path to the new credentials file
    credentials_path = os.path.join(os.path.dirname(__file__), 'BeemoApp', 'GsheetAPI', 'Freq-New-Client.json')
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    creds.refresh(Request())
    try:
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        return sheet
    except HttpError as err:
        logger.error(err)
        return None

# Function to add headers if they don't exist
def add_headers(sheet):
    headers = ["Date", "Time", "Average Frequency", "Highest Frequency", "Lowest Frequency", "Activity Level"]
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = result.get('values', [])
    if not values or values[0] != headers:
        body = {'values': [headers]}
        sheet.values().update(
            spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME,
            valueInputOption="RAW", body=body).execute()

async def record_and_analyze(bot, channel_id, stream, duration=15):
    print("Listening...")

    # Initialize variables
    start_time = time.time()
    frequencies = []
    avg_freq = None
    highest_freq = None
    lowest_freq = None
    activity_level = "No activity captured"
    MIN_FREQUENCY = 20  # Minimum threshold for the lowest frequency

    try:
        # Use sounddevice.InputStream with device info if provided
        device_info = {"channels": CHANNELS, "samplerate": RATE, "dtype": 'int16'}
        if stream is not None and isinstance(stream, tuple) and len(stream) >= 2:
            device_info["device"] = stream[0]
            device_info["channels"] = stream[1]

        with sd.InputStream(**device_info) as audio_stream:
            while time.time() - start_time < duration:
                data, overflowed = audio_stream.read(CHUNK)
                if overflowed:
                    logger.warning("Audio buffer has overflowed")
                
                # Process the audio data
                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    fft_data = np.fft.fft(audio_data)
                    freqs = np.fft.fftfreq(len(fft_data), 1.0 / RATE)
                    idx = np.argmax(np.abs(fft_data))
                    freq = abs(freqs[idx])
                    if freq >= MIN_FREQUENCY:
                        frequencies.append(freq)
                
                await asyncio.sleep(0.1)  # Short sleep to prevent busy waiting

        # Calculate statistics
        if frequencies:
            avg_freq = np.mean(frequencies)
            highest_freq = np.max(frequencies)
            lowest_freq = np.min(frequencies)
            
            # Determine activity level based on frequency ranges
            if avg_freq < 100:
                activity_level = "Low Activity"
            elif 100 <= avg_freq <= 300:
                activity_level = "Normal Activity"
            elif 300 < avg_freq <= 500:
                activity_level = "High Activity"
            else:
                activity_level = "Chaotic"  # or "Stressed"

        # Get current date and time
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Connect to Google Sheets and append data
        sheet = connect_to_google_sheets()
        if sheet:
            add_headers(sheet)
            row = [date_str, time_str, 
                str(round(avg_freq, 2)) if avg_freq else "N/A",
                str(round(highest_freq, 2)) if highest_freq else "N/A",
                str(round(lowest_freq, 2)) if lowest_freq else "N/A",
                activity_level]
            body = {'values': [row]}
            sheet.values().append(
                spreadsheetId=SPREADSHEET_ID,
                range=RANGE_NAME,
                valueInputOption="RAW",
                body=body).execute()

        # Send results to Discord
        message = f"Recording Analysis:\nDate: {date_str}\nTime: {time_str}\n"
        message += f"Average Frequency: {round(avg_freq, 2) if avg_freq else 'N/A'} Hz\n"
        message += f"Highest Frequency: {round(highest_freq, 2) if highest_freq else 'N/A'} Hz\n"
        message += f"Lowest Frequency: {round(lowest_freq, 2) if lowest_freq else 'N/A'} Hz\n"
        message += f"Activity Level: {activity_level}"
        
        await bot.get_channel(channel_id).send(message)
        return activity_level

    except Exception as e:
        logger.error(f"An error occurred during recording and spectrogram generation: {str(e)}")
        if bot and channel_id:
            await bot.get_channel(channel_id).send(f"Error: {str(e)}")
        raise

# Function to send a message to Discord
async def send_discord_message(bot, channel_id, message):
    channel = bot.get_channel(channel_id)
    if channel:
        await channel.send(message)
