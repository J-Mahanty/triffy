from discord.ext import commands 
from dotenv import load_dotenv

import discord
import os


load_dotenv()

server_bot_token = os.getenv('SERVER_BOT_TOKEN')
channel_id = int (os.getenv('CHANNEL_ID'))

client = commands.Bot(command_prefix= '!', intents=discord.Intents.all())


@client.event
async def on_ready(): 
    print("Triffy is now online")
    try:
        channel = await client.fetch_channel(channel_id)
        await channel.send("Triffy is now online")
    except discord.NotFound:
        print("Incorrect Channel ID")
    except discord.Forbidden:
        print("Triffy Does not have permission to view this")
    except Exception as e:
        print(f"An error occurred: {e}")


@client.event
async def on_member_join(member) :
    channel = client.get_channel(channel_id)
    await channel.send(f"Welcome to your Traffic Management app {member.mention}! We will dm you shortly with more.")


@client.command() 
async def hello(ctx) :
    await ctx.send(f"Hello {ctx.author.name}, I am triffy")

client.run(server_bot_token)