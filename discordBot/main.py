from discord.ext import commands 
from dotenv import load_dotenv
from datetime import datetime 
from zoneinfo import ZoneInfo

import discord
import os
import asyncio


load_dotenv()

server_bot_token = os.getenv('SERVER_BOT_TOKEN')
channel_id = int (os.getenv('CHANNEL_ID'))

client = commands.Bot(command_prefix= '!', intents=discord.Intents.all())


@client.event
# On the bot's Startup 
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
# When a new member joins
async def on_member_join(member) :
    #Send hello greeting in server
    channel = client.get_channel(channel_id)
    await channel.send(f"Welcome to your Traffic Management app {member.mention}! We will dm you shortly with more.")

    #Send Dm to member
    try:
        user = await client.fetch_user(member.id)
        await channel.send(f"Attempting to direct message {member.name}")
        await user.send(f"Hello {user.name}! This is the start of your chat with Triffy. Say something to get started.")
    except discord.Forbidden:
        # If permission denied
        await channel.send(f"Could not DM {member.name}. They have DMs disabled.")
    except Exception as e:
        print(f"An error occurred while DMing: {e}")


@client.command(help="Say hello to Triffy") 
# Reply to hello greeting
async def hello(ctx) :
    await ctx.send(f"Hello {ctx.author.name}, I am triffy")


@client.command(help="Start a new trip")
# Information about the Origin, Destination and Time of travel. 
async def route(ctx):
    
    await ctx.send("Current Location?")

    # Making sure bot responds in the same channel, with the same user
    def check(message):
        return message.author == ctx.author and message.channel == ctx.channel

    try:
        origin_message = await client.wait_for("message", check=check, timeout=120)
        origin = origin_message.content

        await ctx.send("Destination?")
        destination_message = await client.wait_for("message", check=check, timeout=120)
        destination = destination_message.content

        current_time = datetime.now(ZoneInfo("Asia/Kolkata"))

        await ctx.send(
            f"**Trip saved!**\n\n"
            f"**Origin:** {origin}\n"
            f"**Destination:** {destination}\n"
            f"**Started at:** {current_time.strftime('%I:%M %p')}"
        )

    except asyncio.TimeoutError:
        await ctx.send("You took too long to respond. Please use !route again.")


@client.event
# Error Handling
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        command_list = "\n".join(
            f"`!{command.name}` - {command.help or 'No description available'}"
            for command in client.commands
        )

        await ctx.send(
            "Invalid command!\n\n"
            "**Available commands:**\n"
            f"{command_list}"
        )
        return 
    raise error


client.run(server_bot_token)