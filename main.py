import os
import io
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

# Import the SDK and types from google-genai
from google import genai
from google.genai import types

# Load environment variables and API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY not found in environment")

# Initialize the client and set the model
client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.0-flash-exp"

app = FastAPI()

# Pydantic model for generating content
class GenerateRequest(BaseModel):
    contents: str

@app.post("/generate")
async def generate_content(request: GenerateRequest):
    """
    Generates an image and text response from a text prompt.
    Returns the text response and image as a base64-encoded string.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=request.contents,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        result_text = None
        image_base64 = None
        for part in response.candidates[0].content.parts:
            if part.text is not None and result_text is None:
                result_text = part.text
            elif part.inline_data is not None and image_base64 is None:
                image_base64 = base64.b64encode(part.inline_data.data).decode("utf-8")
        return {"text": result_text, "image_base64": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic model for editing images
class EditRequest(BaseModel):
    prompt: str

@app.post("/edit")
async def edit_image(prompt: str = Form(...), file: UploadFile = File(...)):
    """
    Edits an uploaded image based on the provided prompt.
    Returns the edited image as a base64-encoded string and any text response.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                prompt,
                image
            ],
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        result_text = None
        image_base64 = None
        for part in response.candidates[0].content.parts:
            if part.text is not None and result_text is None:
                result_text = part.text
            elif part.inline_data is not None and image_base64 is None:
                image_base64 = base64.b64encode(part.inline_data.data).decode("utf-8")
        return {"text": result_text, "image_base64": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic model for chat messages
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_message(request: ChatRequest):
    """
    Creates a new chat session, sends a message, and returns the response.
    Note: For persistent sessions, consider session management.
    """
    try:
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        response = chat.send_message(request.message)
        result_text = None
        image_base64 = None
        for part in response.candidates[0].content.parts:
            if part.text is not None and result_text is None:
                result_text = part.text
            elif part.inline_data is not None and image_base64 is None:
                image_base64 = base64.b64encode(part.inline_data.data).decode("utf-8")
        return {"text": result_text, "image_base64": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
