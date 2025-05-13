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

def find_collection(obj_id, obj_content, collection_dict, tp):
    for abs_id in collection_dict.keys():
        if int(obj_id) in collection_dict[abs_id]["id"]:
            if int(obj_id) not in collection_dict[abs_id]["recorded_id"]:
                collection_dict[abs_id]["recorded_id"].append(int(obj_id))
                obj_content["id"] = int(obj_id)
                obj_content["tp"] = tp
                collection_dict[abs_id]["frames"].append(obj_content)




    # print(f"Saved masked image to: {output_path}")


def main():
    # root_dir = "/home/zl3466/Documents/github/SuperX"
    # data_dir = "/home/zl3466/Documents/dataset/SuperX/mini"
    # out_dir = f"{root_dir}/outputs"
    root_dir = "/Users/zhihengli/Downloads/SuperX"
    data_dir = f"{root_dir}/mini"
    out_dir = f"{root_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)

    img_dir = f"{data_dir}/images"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum_tiny"
    ann_dir = f"{data_dir}/annotations"

    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    bbox_list = os.listdir(instance_bbox_dir)
    bbox_list = [f for f in bbox_list if f.endswith('.json')]
    bbox_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    ann_list = os.listdir(ann_dir)
    ann_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    last_frame_json = bbox_list[-1]
    last_frame_dict = json.load(open(f"{instance_bbox_dir}/{last_frame_json}"))
    object_id_list = list(last_frame_dict.keys())
    last_frame_dict = {k: last_frame_dict[k] for k in sorted(last_frame_dict.keys(), key=int)}

    # collect obj temporals
    obj_record_dict = {}
    obj_id_abs = 0
    for obj_id in object_id_list:
        obj_record = last_frame_dict[obj_id]
        record_list = obj_record["id"]
        obj_record_dict[str(obj_id_abs)] = {"id": record_list, "recorded_id": [], "frames": []}
        obj_id_abs += 1
    print(f"we have a total of {len(obj_record_dict.keys())} distinct objects")
    # with open(f"{out_dir}/same_obj.json", "w") as f:
    #     json.dump(obj_record_dict, f, indent=4)


    # counter = 0
    for bbox_json in tqdm(bbox_list, desc="Processing bounding boxes"):
        tp = bbox_json.split(".")[0]
    #     ann = np.load(f"{ann_dir}/{tp}.npz")
    #     full_img = cv2.imread(f"{img_dir}/{tp}.png")
    #
    #     labels = ann["labels"]
    #     bboxes = ann["bboxes"]
    #     masks = ann["masks"]
    #     confidences = ann["confidences"]
    #
    #     # print(labels)
    #     # print(bboxes)
    #     # print(masks)
    #     # print(confidences)
    #     obj_mask_viz(tp, full_img, labels, masks, bboxes, confidences, f"{data_dir}/img_seg")

        tp = int(tp[:13]) if len(tp) > 13 else int(tp)

        bbox_data = json.load(open(f"{instance_bbox_dir}/{bbox_json}"))
        object_id_list = list(bbox_data.keys())
        # cluster object records
        for obj_id in object_id_list:
            find_collection(obj_id, bbox_data[obj_id], obj_record_dict, tp)
    print(f"processed {len(bbox_list)} frames of bbox")

    for abs_obj_id in obj_record_dict.keys():
        obj_frames = obj_record_dict[abs_obj_id]["frames"]
        prev_center = obj_frames[0]["center"]
        # attach temporal_relationship
        for i in range(len(obj_record_dict[abs_obj_id]["frames"])):
            # for obj_id in obj_record_dict[abs_obj_id]["recorded_id"]:
            one_frame = obj_frames[i]
            curr_center = one_frame["center"]
            distance = sum((a - b) ** 2 for a, b in zip(curr_center, prev_center)) ** 0.5

            # Check if the distance exceeds the threshold
            if i == 0:
                obj_record_dict[abs_obj_id]["frames"][i]["temporal_relationship"] = "newly appeared"
            elif distance > 0.5:
                obj_record_dict[abs_obj_id]["frames"][i]["temporal_relationship"] = "moved"
            else:
                obj_record_dict[abs_obj_id]["frames"][i]["temporal_relationship"] = "stationary"
            prev_center = curr_center

        # attach spatial_relationship
        for i in range(len(obj_record_dict[abs_obj_id]["frames"])):
            # for obj_id in obj_record_dict[abs_obj_id]["recorded_id"]:
            one_frame = obj_frames[i]

            # get nearby obj
            obj_record_dict[abs_obj_id]["frames"][i]["spatial_relationship"] = {"in": [], "on": [], "next_to": [], "under": []}
            # Check if the distance exceeds the threshold


    obj_record_dict_clean = copy.deepcopy(obj_record_dict)
    for abs_obj_id in obj_record_dict_clean.keys():
        obj_frames = obj_record_dict_clean[abs_obj_id]["frames"]
        del obj_record_dict_clean[abs_obj_id]["id"]
        del obj_record_dict_clean[abs_obj_id]["recorded_id"]
        for one_frame in obj_frames:
            del one_frame["id"]
            del one_frame["center"]
            del one_frame["bbox3d"]

    # TODO： load each image in
    with open(f"{out_dir}/all_obj_collection.json", "w") as f:
        json.dump(obj_record_dict, f, indent=4)
    with open(f"{out_dir}/all_obj_collection_clean.json", "w") as f:
        json.dump(obj_record_dict_clean, f, indent=4)

    # obj_record_dict_clean = json.load(open(f"{out_dir}/all_obj_collection_clean.json"))
    # moved_id_list = []
    # for abs_obj_id in obj_record_dict_clean.keys():
    #     obj_frames = obj_record_dict_clean[abs_obj_id]["frames"]
    #     for one_frame in obj_frames:
    #         if one_frame["has_moved"] and int(abs_obj_id) not in moved_id_list:
    #             moved_id_list.append(int(abs_obj_id))

    # question_list = [
    #     f"I have a dictionary of {len(obj_record_dict_clean.keys())} object records. The keys are object ids."
    #     f"Within each object record, there is a 'frames' list containing one or multiple frames of "
    #     f"the object's appearance. "
    #     f"Each frame contains the object's label, its 'center' location, and a 'has_moved' bool "
    #     f"indicating whether the object has moved or not compared to the previous time it was recorded.\n"
    #     f"Give me the id numbers of the objects that has moved. "
    #     f"Your answer should be one or more id numbers seperated by comma. If no object has moved, answer None.\n"
    #     f"Give your answer between <answer></answer> tags.\n"
    #     f"\n{obj_record_dict_clean}\n"
    # ]
    #
    # for i in range(len(question_list)):
    #     question_list[i] = f"{pre_prompt}{question_list[i]}{post_prompt}"
    #
    # ''' ================== Inference ================== '''
    # ''' upload images to model and ask questions '''
    # model = GeminiModel()
    # results = model.analyze_text(question_list)
    # ''' ================== save questions and answers ================== '''
    # if results:
    #     print("\nAnalysis Summary:")
    #     for question, answer in results.items():
    #         # print(f"\n{question}")
    #         print(f"{answer}\n")
    #         try:
    #             ans_string = extract_answer(answer)
    #             ans_list = [int(x.strip()) for x in ans_string.split(',')]
    #         except ValueError:
    #             print(f"answer was not returned in the correct format")
    #             return
    #
    #         ''' ================== save questions and answers in md and json ================== '''
    #         print("===========")
    #         print(f"GT: {len(moved_id_list)} object moved: {moved_id_list}\n")
    #         print(f"Inference: {len(ans_list)} object moved: {ans_list}\n")
    #         score, percentage = calculate_match_score(moved_id_list, ans_list)
    #         print(f"final score: %{percentage}\n{score} out of {len(moved_id_list)} true positives\n")


if __name__ == "__main__":
    main()
