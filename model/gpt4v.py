import hashlib
import json
import os
import base64
from datetime import datetime

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
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _generate_session_id(self, image_paths):
        """Generate a unique session ID based on image paths and modification times."""
        hash_input = ""
        for path in sorted(image_paths):
            if os.path.exists(path):
                # Include file path and last modification time in the hash
                mod_time = os.path.getmtime(path)
                hash_input += f"{path}:{mod_time};"

        # Create a hash of the input string
        return hashlib.md5(hash_input.encode()).hexdigest()

    def analyze_text(self, prompt):
        """Get answers for pure textual input."""
        try:
            print(f"Processing text prompt...")

            # Generate content with text-only prompt
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in GPT-4o text analysis: {str(e)}")
            return None

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
                        # print(f"✓ Loaded: {os.path.basename(path)}")
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

    def analyze_images_new(self, image_paths, prompt, reuse_session=None):
        """Analyze multiple images with a given prompt.

        Args:
            image_paths: List of paths to images
            prompt: Text prompt to send with the images
            reuse_session: Session ID to reuse cached images (optional)
        """
        try:
            # Use provided session ID or generate a new one
            session_id = reuse_session or self._generate_session_id(image_paths)
            cache_file = os.path.join(self.cache_dir, f"{session_id}.json")

            # Check if we can reuse cached images
            image_parts = []
            if reuse_session and os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        image_parts = cached_data.get('image_parts', [])
                        original_paths = cached_data.get('original_paths', [])

                        print(f"Reusing {len(image_parts)} previously uploaded images from session: {reuse_session}")
                        print(f"Original image paths: {', '.join(os.path.basename(p) for p in original_paths)}")
                except Exception as e:
                    print(f"Error loading cached images: {str(e)}")
                    image_parts = []

            # If no cached images or cache loading failed, process the images
            if not image_parts:
                image_parts = []
                failed_images = []
                successful_paths = []

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
                            successful_paths.append(path)
                            print(f"✓ Loaded: {os.path.basename(path)}")
                    except Exception as e:
                        failed_images.append((path, str(e)))

                if failed_images:
                    print("\nFailed to process images:")
                    for path, error in failed_images:
                        print(f"✗ {os.path.basename(path)}: {error}")

                if not image_parts:
                    raise ValueError("No images were successfully processed")

                # Save image data to cache file
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'image_parts': image_parts,
                    'original_paths': successful_paths
                }

                try:
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f)
                    print(f"\nSaved image data to session: {session_id}")
                    print(f"To reuse these images in future queries, use: reuse_session='{session_id}'")
                except Exception as e:
                    print(f"Warning: Failed to cache images: {str(e)}")

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