import logging
import discord
from discord.ext import commands
import asyncio
# import os
from tensorflow.keras.models import load_model
from BNBpredictor import collect_new_data_and_labels as collect_bnb, retrain_model as retrain_bnb_model, save_results_to_google_sheets as save_bnb_results
from QNQpredictor import collect_new_data_and_labels as collect_qnq, retrain_qnq_model, QNQ_save_results_to_google_sheets as save_qnq_results
import os

# Update the image path to be relative
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SavedIMG', 'last_processed_image.png')

# Replace with your actual bot token and channel ID
TOKEN = 'MTMxMzM0ODk1NzkwODM3MzUxNA.GssT8c.nIXJQSlI2UAjCAPSjthdvxZ8kW1djeOyoPEK-k'
CHANNEL_ID = 1311152755276124181  # Ensure this is an integer

# Initialize Discord intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Explicitly enable message content intent

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("discordbot")

# Load models
base_dir = os.path.dirname(os.path.abspath(__file__))
bnb_model_path = os.path.join(base_dir, 'Model', 'updated_model_overall1.keras')
qnq_model_path = os.path.join(base_dir, 'Model', 'QNQupdated_model_overall17.keras')
bnb_model = load_model(bnb_model_path)
qnq_model = load_model(qnq_model_path)

# Global bot
bot = commands.Bot(command_prefix='! ', intents=intents)

@bot.event
async def on_ready():
    logger.info(f'Logged in as {bot.user}')

@bot.command(name='hello_beemo')
async def hello_beemo(ctx):
    await ctx.send("Hi I'm Beemo assigned in Hive 1, your beehive monitoring companion")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.lower() == 'hello beemo':
        await message.channel.send("Hi I'm Beemo assigned in Hive 1, your beehive monitoring companion")

    await bot.process_commands(message)


@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user.name}")
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send("Beemo Dos is connected!")

@bot.command()
async def bnbincorrect(ctx):
    logger.info("Received command: bnbincorrect")
    await ctx.send("Command received: BNB Incorrect. Triggering retraining...")
    await trigger_retraining("bnb", ctx)

@bot.command()
async def qnqincorrect(ctx):
    logger.info("Received command: qnqincorrect")
    await ctx.send("Command received: QNQ Incorrect. Triggering retraining...")
    await trigger_retraining("qnq", ctx)

async def trigger_retraining(model_type, ctx):
    try:
        # Access the last processed file for retraining
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SavedIMG', 'last_processed_image.png')

        # Determine the true label based on the context or other logic
        true_label = await ask_true_label(ctx)

        if model_type == "bnb":
            new_data, new_labels = collect_bnb(true_label, img_path)
            retrain_bnb_model(bnb_model, new_data, new_labels)
            save_bnb_results(img_path, true_label, "Bees Detected", 0.85, 0.9, True)  # Dummy values for illustration
            logger.info("BNB model retrained.")
        elif model_type == "qnq":
            new_data, new_labels = collect_qnq(true_label, img_path)
            retrain_qnq_model(qnq_model, new_data, new_labels)
            save_qnq_results(img_path, true_label, "Queen Detected", 0.85, 0.9, True)  # Dummy values for illustration
            logger.info("QNQ model retrained.")

        await ctx.send(f"{model_type.upper()} model retrained successfully.")
    except Exception as e:
        logger.error(f"Error in retraining {model_type} model: {e}")
        await ctx.send(f"Failed to retrain {model_type.upper()} model. Error: {e}")

async def ask_true_label(ctx):
    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel

    await ctx.send("Please provide the true label for the incorrect prediction:")
    msg = await bot.wait_for('message', check=check)
    try:
        true_label = int(msg.content)
        return true_label
    except ValueError:
        await ctx.send("Invalid label provided. Please provide a numeric label.")
        return await ask_true_label(ctx)
    

# Function to send a message to a Discord channel
async def send_discord_message(message, image_path=None):
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send(message)
        if image_path:
            await channel.send(file=discord.File(image_path))

async def run_discord_bot():
    await bot.start(TOKEN)

if __name__ == "__main__":
    asyncio.run(run_discord_bot())