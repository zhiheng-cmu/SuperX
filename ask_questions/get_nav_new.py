import copy
import sys
import os
import rosbag
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib.cm as cm

# CLIP: match an object with similar feature
# Relational context?
pre_prompt = "Think about the question as if you are a human pondering deeply.\n"
post_prompt = "Give your answer between <answer></answer> tags.\n"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gemini_utils import *
from utils.general_utils import *

def main():
    # root_dir = "/home/zl3466/Documents/github/SuperX"
    # data_dir = "/home/zl3466/Documents/dataset/SuperX/mini"
    root_dir = "/Users/zhihengli/Downloads/SuperX"
    data_dir = f"{root_dir}/mini"
    out_dir = f"{root_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)

    # img_dir = f"{data_dir}/images"
    img_dir = "/Users/zhihengli/Downloads/SuperX/image_mini"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum"
    ann_dir = f"{data_dir}/annotations"

    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    img_path_list = [f"{img_dir}/{filename}" for filename in img_list]
    # bbox_list = os.listdir(instance_bbox_dir)
    # bbox_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # ann_list = os.listdir(ann_dir)
    # ann_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    obj_record_dict_clean = json.load(open(f"{out_dir}/all_obj_collection_clean.json"))

    prefix = (f"I have a dictionary of {len(obj_record_dict_clean.keys())} object records. The keys are object ids. "
              f"Within each object record, there is a list containing one or more frames of the object's appearance. "
              f"Each frame contains the following fields: "
              f"'tp': unix timestamp of the frame of appearance, \n"
              f"'label': the object's semantic label, \n"
              f"'spatial_relationship': a dict of object ids that this object is 'in', 'contain', 'on', 'under', or 'beside'.\n"
              f"'status': a string describing the object's temporal status relative to its previous appearance,"
              f" i.e. new, moved, persistent, disappear. New means this is the first time this object is observed. "
              f"Persistent means the object remained stationary. Disappear means the object was seen at this place before but not anymore. \n"
              )
    suffix = (f"Your answer should be one or more id numbers seperated by comma. "
              f"If there is no object meeting the requirement, answer None.\n"
              f"\n{obj_record_dict_clean}\n")
    # question_list = [
    #     # general abstraction
    #     f"{prefix}"
    #     f"I am looking for an office chair with wheels. Which object is it likely to be?"
    #     f"Your answer should be one or more id numbers seperated by comma. "
    #     f"If there is no object meeting the requirement, answer None.\n"
    #     f"\n{obj_record_dict_clean}\n",
    #
    #     # spatial
    #     f"{prefix}"
    #     f"I am sitting on the sofa. Is there a cup I can grab without standing up? "
    #     f"Your answer should be one id number. "
    #     f"If there is no object meeting the requirement, answer None.\n"
    #     f"\n{obj_record_dict_clean}\n",
    #
    #     f"{prefix}"
    #     f"I am running late to a meeting and I need to grab a laptop and a cup with me. Which laptop and cup should I get? "
    #     f"Your answer should be two id numbers. "
    #     f"If there is no object meeting the requirement, answer None.\n"
    #     f"\n{obj_record_dict_clean}\n",
    #
    #     # temporal
    #     f"{prefix}"
    #     f"I left my cup next to two boxes on a table but it is not there anymore. What is the id of my cup? "
    #     f"Your answer should be one id number. "
    #     f"If there is no object meeting the requirement, answer None.\n"
    #     f"\n{obj_record_dict_clean}\n",
    #
    #     f"{prefix}"
    #     f"I left my cup next to two boxes on a table but it is not there anymore. Where is my cup now? "
    #     f"Give me the id of the object where I should look for my cup. "
    #     f"Your answer should be one id number. "
    #     f"If there is no object meeting the requirement, answer None.\n"
    #     f"\n{obj_record_dict_clean}\n",
    #
    #     f"{prefix}"
    #     f"Something dropped out of my backpack sometime before. Where should I try look for the dropped item? "
    #     f"Give me the id of the object(s) where I should look for the item. "
    #     f"Your answer should be one or more id numbers seperated by comma. "
    #     f"If there is no object meeting the requirement, answer None.\n"
    #     f"\n{obj_record_dict_clean}\n"
    #
    # ]

    question_list = [

        # instance level, spatial consistency?
        # answer: chair 368
        f"I am sitting on a sofa chair and there's a painting hanging on the wall above my head. Which chair am I sitting on?\n",

        # answer: chair 118
        f"I am sitting on a sofa chair facing a wall with paintings. Which sofa chair am I sitting on?\n"
    ]

    video_question_list = copy.deepcopy(question_list)
    scene_graph_question_list = copy.deepcopy(question_list)
    for i in range(len(question_list)):
        scene_graph_question_list[i] = f"{pre_prompt}{prefix}{question_list[i]}{suffix}{post_prompt}"



    gt_ans = ["2, 16", "8, 11", "16", "table 4"]
    ''' ================== Inference ================== '''
    ''' upload images to model and ask questions '''
    model = GeminiModel()

    ''' ================== show questions and answers ================== '''
    print(len(img_path_list))
    img_path_list = downsample_frames(img_path_list, original_hz=10, target_hz=2)
    print(len(img_path_list))

    scene_graph_results = model.analyze_text(scene_graph_question_list)
    video_results = model.analyze_images(img_path_list, video_question_list, out_dir, batches=None)

    if scene_graph_results:
        print("\nAnalysis Summary:")
        for question, answer in scene_graph_results.items():
            # print(f"\n{question}")
            print(f"{answer}\n")

    if video_results:
        print("\nAnalysis Summary:")
        for question, answer in video_results.items():
            # print(f"\n{question}")
            print(f"{answer}\n")

    # ''' ================== show questions and answers ================== '''
    # results = model.analyze_text(question_list)
    # if results:
    #     print("\nAnalysis Summary:")
    #     for question, answer in results.items():
    #         # print(f"\n{question}")
    #         print(f"{answer}\n")
    #         # try:
    #         #     ans_string = extract_answer(answer)
    #         #     ans_list = [int(x.strip()) for x in ans_string.split(',')]
    #         # except ValueError:
    #         #     print(f"answer was not returned in the correct format")
    #         #     return
    #
    #         # ''' ================== save questions and answers in md and json ================== '''
    #         # print("===========")
    #         # print(f"Inference: {len(ans_list)} object moved: {ans_list}\n")
    #
    #     print(f"gt ans {gt_ans}")


if __name__ == "__main__":
    main()
