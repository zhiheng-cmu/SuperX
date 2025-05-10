import copy
import sys
import os
import rosbag
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib.cm as cm

# TODO: use llm to generate relational context text?
#  e.g. query: find me a cup for coffee
# CLIP: match an object with similar feature
# Relational context?
pre_prompt = "Think about the question as if you are a human pondering deeply.\n"
post_prompt = "Give your answer between <answer></answer> tags.\n"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gemini_utils import *


def generate_ranges(total_length, clip_size=20):
    ranges = []
    start = 0
    while start < total_length:
        end = min(start + clip_size, total_length)
        ranges.append([start, end])
        start = end
    return ranges


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def find_collection(obj_id, obj_content, collection_dict):
    for abs_id in collection_dict.keys():
        if int(obj_id) in collection_dict[abs_id]["id"]:
            if int(obj_id) not in collection_dict[abs_id]["recorded_id"]:
                collection_dict[abs_id]["recorded_id"].append(int(obj_id))
                obj_content["id"] = int(obj_id)
                collection_dict[abs_id]["frames"].append(obj_content)

def calculate_match_score(list1, list2):
    """
    Calculate a match score based on the following rule:
    - +1 if an id in list2 is also in list1
    - -1 if an id in list2 is not in list1

    Args:
        list1 (list): Reference list of integers
        list2 (list): List of integers to compare against list1

    Returns:
        int: Total score
        float: Score as a percentage of the maximum possible score
    """
    if not list2:  # If list2 is empty
        return 0, 0.0  # No score possible

    # Convert list1 to a set for faster lookup
    set1 = set(list1)

    score = 0

    for item in list2:
        if item in set1:
            score += 1  # +1 if item is in list1
        else:
            score -= 1  # -1 if item is not in list1
            print("one false positive!")

    max_possible_score = len(list1)
    percentage = score / max_possible_score * 100
    return score, percentage


def obj_mask_viz(tp, full_img, labels, masks, bboxes, confidences, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    output_img = full_img.copy()
    n_objects = len(labels)

    # Use the 'tab20' colormap for up to 20 colors, or 'hsv' for more
    colormap = cm.hsv

    # Generate colors and convert to BGR for OpenCV
    colors = []
    for i in range(n_objects):
        # Generate color from colormap (normalized to [0, 1])
        rgb = colormap(i / max(1, n_objects - 1))[:3]
        # Convert to BGR (OpenCV format) and scale to [0, 255]
        bgr = tuple(int(c * 255) for c in rgb[::-1])
        colors.append(bgr)

    # Create a mask overlay for all objects
    mask_overlay = np.zeros_like(full_img)

    # Project each mask onto the image
    for i in range(len(labels)):
        label = labels[i]
        mask = masks[i]
        bbox = bboxes[i]
        confidence = confidences[i]

        # Get color for this object
        color = colors[i]

        # Create colored mask
        colored_mask = np.zeros_like(full_img)
        colored_mask[mask > 0] = color

        # Add to the overlay
        mask_overlay = np.where(mask[..., np.newaxis] > 0, colored_mask, mask_overlay)

        # Draw bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

        # Add label text
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(output_img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Combine the original image with the mask overlay
    alpha = 0.5  # Transparency factor
    output_img = cv2.addWeighted(output_img, 1, mask_overlay, alpha, 0)

    # Save the output image
    output_path = f"{out_dir}/{tp}_masked.png"
    cv2.imwrite(output_path, output_img)

    # print(f"Saved masked image to: {output_path}")


def main():
    # root_dir = "/home/zl3466/Documents/github/SuperX"
    # data_dir = "/home/zl3466/Documents/dataset/SuperX/mini"
    root_dir = "/Users/zhihengli/Downloads/SuperX"
    data_dir = f"{root_dir}/mini"
    out_dir = f"{root_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)

    img_dir = f"{data_dir}/images"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum"
    ann_dir = f"{data_dir}/annotations"

    # img_list = os.listdir(img_dir)
    # img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # bbox_list = os.listdir(instance_bbox_dir)
    # bbox_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # ann_list = os.listdir(ann_dir)
    # ann_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    obj_record_dict_clean = json.load(open(f"{out_dir}/all_obj_collection_clean.json"))

    prefix = (f"I have a dictionary of {len(obj_record_dict_clean.keys())} object records. The keys are object ids. "
              f"Within each object record, there is a 'frames' list containing one or multiple frames of "
              f"the object's appearance. "
              f"Each frame contains the following fields: "
              f"'label': the object's semantic label, \n"
              f"'tp': unix timestamp of the frame of appearance, \n"
              f"'temporal_relationship': a string describing the object's temporal status relative to its previous appearance,"
              f" i.e. newly appeared, moved, stationary, \n"
              f"'spatial_relationship': a dict of object ids that this object is 'in', 'on', 'next to', or 'under'.\n")
    question_list = [
        f"{prefix}"
        f"I am looking for an office chair with wheels. Which object is it likely to be?"
        f"Your answer should be one or more id numbers seperated by comma. "
        f"If there is no object meeting the requirement, answer None.\n"
        f"\n{obj_record_dict_clean}\n",

        # spatial
        f"{prefix}"
        f"I am sitting on the sofa. Is there a cup I can grab without standing up? "
        f"Your answer should be one id number. "
        f"If there is no object meeting the requirement, answer None.\n"
        f"\n{obj_record_dict_clean}\n",

        f"{prefix}"
        f"I am running late to a meeting and I need to grap a laptop and a cup with me. Which laptop and cup should I get? "
        f"Your answer should be two id numbers. "
        f"If there is no object meeting the requirement, answer None.\n"
        f"\n{obj_record_dict_clean}\n",

        # temporal
        f"{prefix}"
        f"I left my cup next to two boxes on a table but it is not there anymore. What is the id of my cup? "
        f"Your answer should be one id number. "
        f"If there is no object meeting the requirement, answer None.\n"
        f"\n{obj_record_dict_clean}\n",

        f"{prefix}"
        f"I left my cup next to two boxes on a table but it is not there anymore. Where is my cup now? "
        f"Give me the id of the object where I should look for my cup. "
        f"Your answer should be one id number. "
        f"If there is no object meeting the requirement, answer None.\n"
        f"\n{obj_record_dict_clean}\n",

        f"{prefix}"
        f"Something dropped out of my backpack sometime before. Where should I try look for the dropped item? "
        f"Give me the id of the object(s) where I should look for the item. "
        f"Your answer should be one or more id numbers seperated by comma. "
        f"If there is no object meeting the requirement, answer None.\n"
        f"\n{obj_record_dict_clean}\n",

    ]

    for i in range(len(question_list)):
        question_list[i] = f"{pre_prompt}{question_list[i]}{post_prompt}"




    gt_ans = ["2, 16", "8, 11", "16", "table 4"]
    ''' ================== Inference ================== '''
    ''' upload images to model and ask questions '''
    model = GeminiModel()
    results = model.analyze_text(question_list)
    ''' ================== save questions and answers ================== '''
    if results:
        print("\nAnalysis Summary:")
        for question, answer in results.items():
            # print(f"\n{question}")
            print(f"{answer}\n")
            # try:
            #     ans_string = extract_answer(answer)
            #     ans_list = [int(x.strip()) for x in ans_string.split(',')]
            # except ValueError:
            #     print(f"answer was not returned in the correct format")
            #     return

            # ''' ================== save questions and answers in md and json ================== '''
            # print("===========")
            # print(f"Inference: {len(ans_list)} object moved: {ans_list}\n")

        print(f"gt ans {gt_ans}")


if __name__ == "__main__":
    main()
