import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

class GPT4VModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
    def analyze_images(self, image_paths, prompt):
        """Analyze multiple images with a given prompt."""
        try:
            # Process images
            image_parts = []
            failed_images = []
            
            for path in image_paths:
                try:
                    if not os.path.exists(path):
                        failed_images.append((path, "File not found"))
                        continue
                        
                    with open(path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                        print(f"✓ Loaded: {os.path.basename(path)}")
                except Exception as e:
                    failed_images.append((path, str(e)))
            
            if failed_images:
                print("\nFailed to process images:")
                for path, error in failed_images:
                    print(f"✗ {os.path.basename(path)}: {error}")
            
            if not image_parts:
                raise ValueError("No images were successfully processed")
            
            print(f"\nAnalyzing {len(image_parts)} images...")
            
            # Generate content with specific configuration
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *image_parts
                        ]
                    }
                ],
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in GPT-4o analysis: {str(e)}")
            return None