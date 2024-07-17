import os
from io import BytesIO

import chainlit as cl
import pandas as pd
from chainlit import on_chat_start, on_audio_chunk, on_audio_end
from chainlit.element import ElementBased
from dotenv import load_dotenv
from pandasai import SmartDataframe, Agent
from pandasai.llm import OpenAI

from openai import AsyncOpenAI

load_dotenv()

cl.instrument_openai()
client = AsyncOpenAI()


@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


@on_chat_start
async def main():
    files = None
    llm = OpenAI()

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a csv to begin!", accept=["text/csv"]
        ).send()

    text_file = files[0]

    data = pd.read_csv(text_file.path)
    pandas_ai = Agent(data)

    cl.user_session.set("agent", pandas_ai)

    # Let the user know that the system is ready
    await cl.Message(
        content=f"`{text_file.name}` uploaded!"
    ).send()


@on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements]
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    pandas_ai = cl.user_session.get("agent")
    response = pandas_ai.chat(transcription)
    print(response)
    await cl.Message(
        content=response,
    ).send()


@on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # TODO: Use Gladia to transcribe chunks as they arrive would decrease latency
    # see https://docs-v1.gladia.io/reference/live-audio

    # For now, write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)
