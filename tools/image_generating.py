from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
import base64
import json
import logging
from config import get_client  

def generate_or_edit_image(prompt, image_path=None, output_path='output.png'):
    try:
        # Get client instance
        client = get_client()
        if client is None:
            return {
                "success": False,
                "message": "Failed to initialize Gemini client",
                "error": "Client initialization failed"
            }

        # Validate inputs
        if not prompt or not isinstance(prompt, str):
            return {
                "success": False,
                "message": "Invalid prompt: Must be a non-empty string",
                "error": "Invalid prompt"
            }

        # If image_path is provided and the file exists, perform image editing
        if image_path and os.path.isfile(image_path):
            try:
                image = Image.open(image_path)
                contents = [prompt, image]
            except Exception as img_error:
                return {
                    "success": False,
                    "message": f"Error opening input image: {str(img_error)}",
                    "error": str(img_error)
                }
        else:
            # Otherwise, perform image generation from prompt only
            contents = prompt

        # Make API call
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            logging.info(f"API Response: {response}")
        except Exception as api_error:
            return {
                "success": False,
                "message": f"API call failed: {str(api_error)}",
                "error": str(api_error)
            }

        # Validate response
        if not hasattr(response, 'candidates') or not response.candidates:
            return {
                "success": False,
                "message": "No response received from the model",
                "error": "Empty response"
            }

        # Process response
        image_created = False
        generated_text = ""
        image_data = None

        try:
            for part in response.candidates[0].content.parts:
                logging.info(f"Processing part: {part}")
                if hasattr(part, 'text') and part.text is not None:
                    generated_text += part.text + "\n"
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    try:
                        image = Image.open(BytesIO(part.inline_data.data))
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                        image.save(output_path)
                        # Convert image to base64
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        image_data = base64.b64encode(buffered.getvalue()).decode()
                        image_created = True
                        logging.info("Image successfully created and encoded")
                    except Exception as img_error:
                        logging.error(f"Error processing image data: {str(img_error)}")
                        return {
                            "success": False,
                            "message": f"Error processing image data: {str(img_error)}",
                            "error": str(img_error)
                        }
        except Exception as process_error:
            return {
                "success": False,
                "message": f"Error processing response: {str(process_error)}",
                "error": str(process_error)
            }

        if image_created:
            result = {
                "success": True,
                "message": f"IMAGE_GENERATED: Image successfully created and saved to {output_path}",
                "output_path": output_path,
                "generated_text": generated_text.strip(),
                "image_data": f"data:image/png;base64,{image_data}"
            }
            logging.info(f"Returning result: {json.dumps(result, indent=2)}")
            return result
        else:
            return {
                "success": False,
                "message": "No image was generated in the response",
                "generated_text": generated_text.strip()
            }

    except Exception as e:
        logging.error(f"Unexpected error in generate_or_edit_image: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Error occurred: {str(e)}",
            "error": str(e)
        }
