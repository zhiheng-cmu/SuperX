import copy
import sys
import os
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
    video_ori_freq = 10
    video_target_freq = 1
    gpt_reuse_session = '11a590587e4e113e91e715cf74332c8f'

    # root_dir = "/Users/zhihengli/Downloads/SuperX"
    root_dir = "/home/zl3466/Documents/dataset/SuperX"
    data_dir = f"{root_dir}/tracking"
    out_dir = f"{data_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)

    # img_dir = f"{data_dir}/images"
    img_dir = f"{data_dir}/image"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum"
    ann_dir = f"{data_dir}/annotations"

    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    img_path_list = [f"{img_dir}/{filename}" for filename in img_list]
    # bbox_list = os.listdir(instance_bbox_dir)
    # bbox_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # ann_list = os.listdir(ann_dir)
    # ann_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    obj_record_dict_clean = json.load(open(f"{out_dir}/scene_graph_clean.json"))

    scene_graph_prefix = (
        f"I have a dictionary of {len(obj_record_dict_clean.keys())} object records. The keys are object ids. "
        f"Some of these ids may refer to the same object."
        f"Within each object record, there is a list containing one or more frames of the object's appearance. "
        f"Each frame contains the following fields: "
        f"'tp': unix timestamp of the frame of appearance, \n"
        f"'label': the object's semantic label, \n"
        f"'center': the object's center location [x, y, z] in global coordinate frame with z representing elevation, \n"
        # f"'spatial_relationship': a dict of object ids that this object is 'in', 'contain', 'on', 'under', or 'beside'.\n"
        f"'status': a string describing the object's temporal status relative to its previous appearance,"
        f" i.e. new, moved, persistent, disappeared. New means this is the first time this object is observed. "
        f"Persistent means the object remained stationary. Disappear means the object was seen at this place before but not seen upon revisit. \n"
        )
    scene_graph_suffix = (f"Give your answer as one or more object id numbers seperated by comma. "
                          f"If there is no object meeting the requirement, answer None.\n"
                          f"\n{obj_record_dict_clean}\n")

    video_prefix = f"This is a 360-degree video recorded at {video_target_freq} Hz from a robot.\n"
    video_suffix = f"\n"

    question_list = [

        # Spatial Consistency: instance-level object retrieval based on spatial relations
        f"I am at the cone, and the plant on the table is on fire. I need to put out the fire asap. Which fire extinguisher should I use?\n",

        # Temporal Consistency: given past state, inquire current state
        f"I left my jacket on a chair next to the plant on the table a few minutes ago, but it is not there anymore. Where is my jacket now? \n",

        # Memory-Based Retrieval: Given current state, inquire sequence of past state
        # demonstrate:
        #   1. instance level retrieval (get my bag out of multiple bags present in the scene)
        #   2. memory of historical event (obj trajectory / past obj location )
        f"Something dropped out of my bag. It must have dropped as I carried my bag around. "
        f"Where should I look for the dropped item? \n"
    ]

    video_question_list = copy.deepcopy(question_list)
    scene_graph_question_list = copy.deepcopy(question_list)
    for i in range(len(question_list)):
        scene_graph_question_list[i] = f"{pre_prompt}{scene_graph_prefix}{question_list[i]}{scene_graph_suffix}{post_prompt}"
        video_question_list[i] = f"{pre_prompt}{video_prefix}{question_list[i]}{video_suffix}{post_prompt}"


    ''' ================== Inference ================== '''
    ''' upload images to model and ask questions '''
    model = GeminiModel()
    # model = GPT4VModel()

    ''' ================== show questions and answers ================== '''
    # print(len(img_path_list))
    img_path_list = downsample_frames(img_path_list, original_hz=video_ori_freq, target_hz=video_target_freq)
    print("upload video length: ", len(img_path_list))

    print("Analyzing with scene graph...")
    scene_graph_results = model.analyze_text(scene_graph_question_list)
    print("Analyzing with video...")
    # video_results = model.analyze_images(img_path_list, video_question_list, out_dir, batches=None)
    # video_results = model.analyze_images_new(img_path_list, video_question_list, reuse_session=gpt_reuse_session)

    if scene_graph_results:
        print("\nScene Graph Analysis Summary:")
        for question, answer in scene_graph_results.items():
            # print(f"\n{question}")
            print(f"{answer}\n")
    print("\n=======================\n")
    # if video_results:
    #     print("\nVideo Analysis Summary:")
    #     for question, answer in video_results.items():
    #         # print(f"\n{question}")
    #         print(f"{answer}\n")

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
