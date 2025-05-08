import sys
import os
import rosbag
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

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




def main():
    root_dir = "/home/zl3466/Documents/github/SuperX"
    data_dir = "/home/zl3466/Documents/dataset/SuperX/mini"

    img_dir = f"{data_dir}/images"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum"
    bbox_data = json.load(open(f"{instance_bbox_dir}/1736999024763313408.json"))
    # Extract unique labels
    unique_labels = list(set(entry["label"] for entry in bbox_data.values()))
    print(unique_labels)
    out_dir = f"{root_dir}/outputs"
    # space_list = ["hallway"]
    space_list = None
    question_list = [
        f"This is a image from a robot exploring an indoor space. "
        f"The space is known to contain at least the following objects: {unique_labels}.\n"
        f"Determine the room or space it appears to be in.\n"
        f"Your answer should be a single word or phrase for the name of the room or space.\n"
        f"Please provide your text answer within the <answer> </answer> tags.\n\n",

        f"This is a image from a robot exploring an indoor space. Determine if this space belongs to one of "
        f"{space_list}; if not, give me the space it appears to be in "
        f"(e.g. kitchen, bedroom, livingroom, garage, basement, etc.).\n"
        f"Please provide your text answer within the <answer> </answer> tags.\n\n"
    ]

    ''' ================== Inference ================== '''
    ''' upload images to model and ask questions '''
    model = GeminiModel()
    results = model.analyze_single_image(f"{img_dir}/1736999024763313408.png", [question_list[0]])
    ''' ================== save questions and answers ================== '''
    if results:
        print("\nAnalysis Summary:")
        for question, answer in results.items():
            print(f"\n{question}")
            print(f"{answer}")

    ''' ================== save questions and answers in md and json ================== '''


if __name__ == "__main__":
    main()
