from discord.ext import commands 
from dotenv import load_dotenv

import discord
import os

load_dotenv()

server_bot_token = os.getenv('SERVER_BOT_TOKEN')

client = commands.Bot(command_prefix= '!', intents=discord.Intents.all())

@client.event
async def on_ready() : 
    print("Triffy is now online")


@client.command
async def hello(ctx) :
    await ctx.send(f"Hello {ctx.author.name}, I am triffy")

client.run(server_bot_token)