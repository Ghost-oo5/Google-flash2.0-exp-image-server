import os
import io
import base64
from fastapi import FastAPI, HTTPException
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

# Helper function to format base64 image with prefix
def format_base64_image(b64_string: str) -> str:
    return f"data:image/jpeg;base64,{b64_string}"

# Pydantic models defined in the same file
class GenerateRequest(BaseModel):
    contents: str

class EditRequest(BaseModel):
    prompt: str
    image_base64: str

class ChatRequest(BaseModel):
    message: str

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
                image_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                image_base64 = format_base64_image(image_b64)
        return {"text": result_text, "image_base64": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit")
async def edit_image(request: EditRequest):
    """
    Edits an image based on the provided prompt and base64-encoded image data.
    Returns the edited image as a base64-encoded string along with any text response.
    """
    try:
        image_str = request.image_base64
        # Remove the data URL prefix if present
        if image_str.startswith("data:image"):
            image_str = image_str.split(",", 1)[1]
        
        # Decode the base64 image data
        image_data = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_data))

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                request.prompt,
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
                image_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                image_base64 = format_base64_image(image_b64)
        return {"text": result_text, "image_base64": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
                image_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                image_base64 = format_base64_image(image_b64)
        return {"text": result_text, "image_base64": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
