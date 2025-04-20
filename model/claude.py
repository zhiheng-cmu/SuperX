import os
import base64
from anthropic import Anthropic
from dotenv import load_dotenv

class ClaudeModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = Anthropic(api_key=api_key)
        
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
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
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
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *image_parts
                        ]
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"Error in Claude analysis: {str(e)}")
            return None