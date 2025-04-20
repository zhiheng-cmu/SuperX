import base64
import os
import json
import re
import sys
import time
from datetime import datetime
from google import genai
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm


GEMINI_MODEL = "gemini-2.0-flash"


def check_uploaded_file(file_path):
    creation_time = os.path.getctime(file_path)
    current_time = time.time()

    # Check if the file is older than 2 days (2 * 24 * 60 * 60 seconds)
    if (current_time - creation_time) > (2 * 86400):
        os.remove(file_path)
        print(f"The images have been uploaded but already expired, removing existing filename records: {file_path}")
        return False
    else:
        print(f"The images have been uploaded and are still valid on Gemini: {file_path}")
        return True

class GeminiModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        # print(api_key)
        self.client = genai.Client(api_key=api_key)
        self.model = self.client.models
        # self.model = genai.GenerativeModel(GEMINI_MODEL)
        
    def analyze_images(self, image_paths, prompt_list, out_dir, batches=None):
        """Analyze multiple images with a given prompt."""
        filename_output_json = f"{out_dir}/uploaded_filename_list_{len(image_paths)}.json"

        # file_dict = {}
        upload_filelist = []
        failed_images = []
        if os.path.exists(filename_output_json):
            valid = check_uploaded_file(filename_output_json)
            if valid:
                upload_filename_list = json.load(open(filename_output_json))
            else:
                upload_filename_list = []
        else:
            upload_filename_list = []
        try:
            if len(upload_filename_list) == len(image_paths):
                for filename in tqdm(upload_filename_list, desc="files already uploaded, retrieving them from remote..."):
                    myfile = self.client.files.get(name=filename)
                    upload_filelist.append(myfile)
            else:
                for path in tqdm(image_paths, desc=f"uploading {len(image_paths)} files for the first time..."):
                    try:
                        if not os.path.exists(path):
                            failed_images.append((path, "File not found"))
                            continue

                        myfile = self.client.files.upload(file=path)
                        upload_filelist.append(myfile)
                        uploaded_file_name = myfile.name
                        upload_filename_list.append(uploaded_file_name)
                        # file_dict[myfile.name] = path

                    except Exception as e:
                        failed_images.append((path, str(e)))
                # print(file_dict)
                ''' files uploaded to gemini have a new token-like filename, save these to local to access later '''
                print(f"saving uploaded filenames to {filename_output_json}")
                with open(filename_output_json, 'w') as f:
                    json.dump(upload_filename_list, f, indent=2)

                if failed_images:
                    print("\nFailed to process images:")
                    for path, error in failed_images:
                        print(f"✗ {os.path.basename(path)}: {error}")

                if not upload_filelist:
                    raise ValueError("No images were successfully processed")
            
            print(f"\nAnalyzing {len(upload_filelist)} images...")
            
            # Generate content with specific configuration
            result = {}
            if batches is not None:
                for batch_idx_pair in batches:
                    filelist = upload_filelist[batch_idx_pair[0]:batch_idx_pair[1]]
                    print(filelist)
                    for prompt_idx in range(len(prompt_list)):
                        prompt = prompt_list[prompt_idx]
                        print(f"analyzing prompt {prompt_idx}/{len(prompt_list)} with images {batch_idx_pair[0]}:{batch_idx_pair[1]}...")
                        response = self.model.generate_content(
                            model=GEMINI_MODEL,
                            contents=[
                                filelist,
                                prompt
                            ]
                        )
                        if prompt not in result.keys():
                            result[prompt] = [response.text]
                        else:
                            result[prompt].append(response.text)
            else:
                for prompt_idx in range(len(prompt_list)):
                    prompt = prompt_list[prompt_idx]
                    print(f"analyzing prompt {prompt_idx}/{len(prompt_list)} with the full set...")
                    response = self.model.generate_content(
                        model=GEMINI_MODEL,
                        contents=[
                            upload_filelist,
                            prompt
                        ]
                    )
                    result[prompt] = response.text
                # print(f"Done analyzing, waiting for 20s to avoid exhausting gemini api quota per minute")
                # for remaining in range(20, 0, -1):
                #     sys.stdout.write(f"\rWaiting: {remaining} seconds...")  # This line is updated
                #     sys.stdout.flush()  # Only flushes this line
                #     time.sleep(1)
                # sys.stdout.write("\rDone!                           \n")  # Clear the line and print 'Done!'

            return result
            
        except Exception as e:
            print(f"Error in Gemini analysis: {str(e)}")
            return None


    @staticmethod
    def extract_json_from_text(text):
        """Extract JSON data from response text."""
        if not text:
            return None
            
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code blocks
            json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
                    
            # Try finding any JSON-like structure
            json_pattern = r'\{[^}]*\}'
            matches = re.findall(json_pattern, text)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    pass
                    
        return None

    @staticmethod
    def save_json(data, prefix="spatial_data"):
        """Save data to a JSON file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        return filename