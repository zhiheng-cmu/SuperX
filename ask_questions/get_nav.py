import sys
import os
import rosbag
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


# TODO: use llm to generate relational context text?
#  e.g. query: find me a cup for coffee
# CLIP: match an object with similar feature
# Relational context?


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
    out_dir = f"{root_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)

    img_dir = f"{data_dir}/images"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum"

    bbox_list = os.listdir(instance_bbox_dir)
    bbox_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    sample_rate = 1  # Hz
    all_obj_dict = {}
    prev_tp = None
    counter = 0
    for bbox_json in tqdm(bbox_list, desc="Processing bounding boxes"):
        tp = bbox_json.split(".")[0]
        tp = int(tp[:13]) if len(tp) > 13 else int(tp)

        if prev_tp is None:
            prev_tp = tp
        else:
            tp_diff_seconds = (tp - prev_tp) / 1000
            if tp_diff_seconds >= (1 / sample_rate) * 0.95:
                prev_tp = tp  # Update the previous timestamp
            else:
                continue
        counter += 1
        bbox_data = json.load(open(f"{instance_bbox_dir}/{bbox_json}"))
        object_id_list = list(bbox_data.keys())
        for obj_id in object_id_list:
            if obj_id not in all_obj_dict.keys():
                all_obj_dict[obj_id] = {
                    "label": bbox_data[obj_id]["label"],
                    "center": bbox_data[obj_id]["center"]
                }
            else:
                obj_center = np.array(bbox_data[obj_id]["center"])
                old_center = np.array(all_obj_dict[obj_id]["center"])
                all_obj_dict[obj_id] = {
                    "label": bbox_data[obj_id]["label"],
                    "center": np.mean([obj_center, old_center], axis=0).tolist()
                }
    print(f"processed {counter} frames of bbox")
    all_obj_dict = {k: all_obj_dict[k] for k in sorted(all_obj_dict.keys(), key=int)}
    with open(f"{out_dir}/all_obj.json", "w") as f:
        json.dump(all_obj_dict, f, indent=4)


    unique_labels = list(set(entry["label"] for entry in all_obj_dict.values()))
    print(unique_labels)

    # # space_list = ["hallway"]
    # space_list = None
    # question_list = [
    #     f"You are in a space with the following objects:\n {all_obj_dict}.\n"
    #     f"In this dictionary, each key-value pair represents an object with "
    #     f"its semantic label ('label') and location ('center').\n"
    #     f"Your answer should be a single word or phrase for the name of the room or space.\n"
    #     f"Please provide your text answer within the <answer> </answer> tags.\n\n",
    #
    #     f"This is a image from a robot exploring an indoor space. Determine if this space belongs to one of "
    #     f"{space_list}; if not, give me the space it appears to be in "
    #     f"(e.g. kitchen, bedroom, livingroom, garage, basement, etc.).\n"
    #     f"Please provide your text answer within the <answer> </answer> tags.\n\n"
    # ]
    #
    # ''' ================== Inference ================== '''
    # ''' upload images to model and ask questions '''
    # model = GeminiModel()
    # results = model.analyze_single_image(f"{img_dir}/1736999024763313408.png", [question_list[0]])
    # ''' ================== save questions and answers ================== '''
    # if results:
    #     print("\nAnalysis Summary:")
    #     for question, answer in results.items():
    #         print(f"\n{question}")
    #         print(f"{answer}")
    #
    # ''' ================== save questions and answers in md and json ================== '''


if __name__ == "__main__":
    main()
