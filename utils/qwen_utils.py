import os
import re
import math
import json
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # convert numpy array to list
        return super(NumpyEncoder, self).default(obj)



def ask_qwen(model, processor, messages):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text

def analyze_street_view_qwen(image_directory, question_list, surround=False, total_length=-1, batches=None,
                             min_pixels=256*28*28, max_pixels=1280*28*28):
    """Analyze street view images in a directory and generate spatial data"""
    # Get and sort image paths from frames directory
    if surround and os.path.exists(f"{image_directory}/frames_surround"):
        frames_dir = f"{image_directory}/frames_surround"
    else:
        frames_dir = f"{image_directory}/frames"

    if os.path.exists(frames_dir):
        image_directory = frames_dir

    # Get all image paths
    image_paths = []
    for image_file in os.listdir(image_directory):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(image_directory, image_file)
            image_paths.append(full_path)

    if not image_paths:
        print(f"No images found in {image_directory}!")
        return None

    # Sort by frame number in reverse order
    image_paths = sorted(image_paths, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(0)), reverse=True)
    if total_length != -1:
        image_paths = image_paths[:total_length]
    print(f"\nFound {len(image_paths)} images in {image_directory}")

    try:
        # Initialize model and tokenizer
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels,
                                              max_pixels=max_pixels)

        result = {}
        if batches is not None:
            ''' process images in img_paths by batches '''
            for batch_idx_pair in batches:
                filelist = image_paths[batch_idx_pair[0]:batch_idx_pair[1]]
                for q_idx in range(len(question_list)):
                    question = question_list[q_idx]
                    print(
                        f"analyzing prompt {q_idx}/{len(question_list)} with images {batch_idx_pair[0]}:{batch_idx_pair[1]}...")

                    content = []
                    for img_path in filelist:
                        content.append({"type": "image", "image": img_path})
                    content.append({"type": "text",
                                    "text": question})
                    print(f"asking question with {len(content) - 1} images")

                    messages = [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]

                    output_text = ask_qwen(model, processor, messages)
                    if question not in result.keys():
                        result[question] = [output_text]
                    else:
                        result[question].append(output_text)
        else:
            ''' process all images together in img_paths '''
            for question in question_list:
                content = []
                for img_path in image_paths:
                    content.append({"type": "image", "image": img_path})
                content.append({"type": "text",
                                "text": question})

                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]

                output_text = ask_qwen(model, processor, messages)
                result[question] = output_text

        return result, image_paths

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None


def parse_json_from_response(response_text):
    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    data = []
    if json_match:
        json_data_str = json_match.group(1)
        data = json.loads(json_data_str)
    return data