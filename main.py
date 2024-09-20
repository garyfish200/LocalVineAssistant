from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import json
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize the OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get the assistant IDs from environment variables
ASSISTANT_ID_JOHNS_CREEK = os.getenv("ASSISTANT_ID_JOHNS_CREEK")
ASSISTANT_ID_ATLANTA = os.getenv("ASSISTANT_ID_ATLANTA")

# Semaphore to limit concurrent API calls
API_SEMAPHORE = asyncio.Semaphore(1000)  # Adjust this number based on your API rate limits

async def get_assistant_id(assistant_type: str):
    if assistant_type == "johns_creek":
        return ASSISTANT_ID_JOHNS_CREEK
    elif assistant_type == "atlanta":
        return ASSISTANT_ID_ATLANTA
    else:
        raise ValueError("Invalid assistant type")

class UserMessage(BaseModel):
    message: str

class ThreadMessage(BaseModel):
    message: str
    thread_id: str

async def stream_response(thread_id: str, run_id: str):
    full_response = ""
    while True:
        async with API_SEMAPHORE:
            run = await client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
        if run.status == "completed":
            async with API_SEMAPHORE:
                messages = await client.beta.threads.messages.list(thread_id=thread_id)
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            if assistant_messages:
                latest_message = assistant_messages[0]
                if latest_message.content:
                    text_content = next((content for content in latest_message.content if content.type == "text"), None)
                    if text_content:
                        full_response = text_content.text.value
            break
        elif run.status == "failed":
            full_response = "Assistant run failed"
            break
        await asyncio.sleep(0.1)
    return json.dumps({"message": full_response, "thread_id": thread_id})

@app.post("/chat")
async def chat_with_assistant(user_message: UserMessage, x_assistant_type: Optional[str] = Header(None)):
    return await process_chat(user_message.message, assistant_type=x_assistant_type)

@app.post("/chat_existing_thread")
async def chat_with_existing_thread(thread_message: ThreadMessage, x_assistant_type: Optional[str] = Header(None)):
    return await process_chat(thread_message.message, thread_message.thread_id, assistant_type=x_assistant_type)

async def process_chat(message: str, thread_id: str = None, assistant_type: str = None):
    try:
        if thread_id is None:
            # Create a new thread
            async with API_SEMAPHORE:
                thread = await client.beta.threads.create()
                thread_id = thread.id
        
        # Add the user's message to the thread
        async with API_SEMAPHORE:
            await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message
            )

        print("Assistant type:", assistant_type)
        # Get the appropriate assistant ID
        try:
            assistant_id = await get_assistant_id(assistant_type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid assistant type")

        print("Assistant ID:", assistant_id)
        # Run the assistant
        async with API_SEMAPHORE:
            run = await client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_choice={"type": "file_search"}
            )

        response = await stream_response(thread_id, run.id)
        return JSONResponse(content=json.loads(response))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
