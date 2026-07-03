import os
import time
import google.generativeai as genai
from PIL import Image
import dotenv

dotenv.load_dotenv()

# Rate limiting config
MAX_RETRIES = 5
BASE_WAIT_SECONDS = 2  # exponential backoff: 2, 4, 8, 16, 32s

def get_gemini_model():
    """Initializes the Gemini model."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Please set it to use Gemini.")
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash which is perfect for fast multimodal tasks
    return genai.GenerativeModel('gemini-2.5-flash')

gemini_model = None

def describe_image(image_path: str) -> str:
    """
    Extracts a description from the provided image using Gemini.
    Returns an empty string if the image is determined to be non-scientific.
    """
    global gemini_model
    if gemini_model is None:
        try:
            gemini_model = get_gemini_model()
        except Exception as e:
            return f"Failed to initialize Gemini model: {e}"
            
    try:
        raw_image = Image.open(image_path).convert('RGB')
        
        prompt = (
            "You are an image which is taken from a research paper from the domain of Atomic Layer Deposition and you are an helpful assistant for analysing these images."
            "Determine if this image is a scientific image (like a chart, graph, diagram, plot, or scientific photograph). "
            "If it is a scientific image, provide a concise but detailed description of its contents, focusing on the data and scientific meaning. "
            "If it is a non-scientific image (like a generic photo, a logo, or decorative element), simply return 'NON_SCIENTIFIC' and nothing else."
            "Do not include any additional information other than the description."
        )
        
        # Retry loop with exponential backoff for rate limit errors
        for attempt in range(MAX_RETRIES):
            try:
                response = gemini_model.generate_content([prompt, raw_image])
                
                if response and response.text:
                    text = response.text.strip()
                    if text == 'NON_SCIENTIFIC':
                        return ""
                    return text
                return ""
            except Exception as api_err:
                err_str = str(api_err).lower()
                if '429' in err_str or 'resource' in err_str or 'quota' in err_str or 'rate' in err_str:
                    wait = BASE_WAIT_SECONDS * (2 ** attempt)
                    print(f"Rate limited on {image_path}. Waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    print(f"Error processing image {image_path}: {api_err}")
                    return ""
        print(f"Exhausted retries for {image_path}. Skipping.")
        return ""
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return ""

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(describe_image(sys.argv[1]))
    else:
        print("Provide an image path as argument to test image captioning.")
