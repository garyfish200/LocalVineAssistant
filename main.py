from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get the assistant ID from environment variables
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

class UserMessage(BaseModel):
    message: str

class ThreadMessage(BaseModel):
    message: str
    thread_id: str

async def stream_response(thread_id: str, run_id: str):
    full_response = ""
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            assistant_messages = [msg for msg in messages if msg.role == "assistant"]
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
async def chat_with_assistant(user_message: UserMessage):
    return await process_chat(user_message.message)

@app.post("/chat_existing_thread")
async def chat_with_existing_thread(thread_message: ThreadMessage):
    return await process_chat(thread_message.message, thread_message.thread_id)

async def process_chat(message: str, thread_id: str = None):
    try:
        if thread_id is None:
            # Create a new thread
            thread = client.beta.threads.create()
            thread_id = thread.id
        
        # Add the user's message to the thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )

        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )

        response = await stream_response(thread_id, run.id)
        return JSONResponse(content=json.loads(response))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
