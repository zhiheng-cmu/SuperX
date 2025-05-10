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

def clean_spatial_relations(spatial_relations_dict):
    clean_dict = {}
    for relation in spatial_relations_dict.keys():
        if len(spatial_relations_dict[relation]) != 0:
            clean_dict[relation] = spatial_relations_dict[relation]

    return clean_dict

def main():
    root_dir = "/media/zl3466/新加卷/dataset/SuperX"
    data_dir = "/media/zl3466/新加卷/dataset/SuperX/set_0"
    out_dir = f"{data_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)

    img_dir = f"{data_dir}/image"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum"
    # ann_dir = f"{data_dir}/annotations"

    bbox_list = os.listdir(instance_bbox_dir)
    bbox_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # ann_list = os.listdir(ann_dir)
    # ann_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    status_list = []
    obj_frame_dict = {}
    for bbox_json in tqdm(bbox_list, desc="Processing bounding boxes"):
        bbox_data = json.load(open(f"{instance_bbox_dir}/{bbox_json}"))
        obj_list = list(bbox_data.keys())
        # tp = int(bbox_json.split(".")[0])
        for obj_id in obj_list:
            obj_record = bbox_data[obj_id]
            if obj_record["status"] not in status_list:
                status_list.append(obj_record["status"])

            if obj_id not in obj_frame_dict.keys():
                obj_frame_dict[obj_id] = [
                    {
                        "tp": obj_record["latest_stamp"],
                        "id": obj_record["id"],
                        "label": obj_record["label"],
                        "center": obj_record["center"],
                        "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                        "status": obj_record["status"]
                    }
                ]
            else:
                obj_frame_dict[obj_id].append(
                    {
                        "tp": obj_record["latest_stamp"],
                        "id": obj_record["id"],
                        "label": obj_record["label"],
                        "center": obj_record["center"],
                        "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                        "status": obj_record["status"]
                    }
                )


    # sort frame list by tp for each object
    full_obj_list = list(obj_frame_dict.keys())
    for obj_id in full_obj_list:
        frame_list = obj_frame_dict[obj_id]
        obj_frame_dict[obj_id] = sorted(frame_list, key=lambda x: x["tp"])

    # clean up merged object
    clean_obj_list = []
    for obj_id in full_obj_list:
        frame_list = obj_frame_dict[obj_id]
        obj_record = frame_list[-1]
        if obj_record["id"][0] not in clean_obj_list:
            clean_obj_list.append(obj_record["id"][0])

    clean_frame_dict = {}
    for obj_id in clean_obj_list:
        frame_list = obj_frame_dict[obj_id]
        clean_frame_dict[obj_id] = frame_list
        for one_frame in frame_list:
            del one_frame["id"]

    with open(f"{out_dir}/scene_graph_clean.json", "w") as f:
        json.dump(clean_frame_dict, f, indent=4)
    print(status_list)
    # last_frame_json = bbox_list[-1]
    # last_frame_dict = json.load(open(f"{instance_bbox_dir}/{last_frame_json}"))
    # object_id_list = list(last_frame_dict.keys())
    # last_frame_dict = {k: last_frame_dict[k] for k in sorted(last_frame_dict.keys(), key=int)}
    #
    # # collect obj temporals
    # obj_record_dict = {}
    # obj_id_abs = 0
    # for obj_id in object_id_list:
    #     obj_record = last_frame_dict[obj_id]
    #     record_list = obj_record["id"]
    #     obj_record_dict[str(obj_id_abs)] = {"id": record_list, "recorded_id": [], "frames": []}
    #     obj_id_abs += 1
    # print(f"we have a total of {len(obj_record_dict.keys())} distinct objects")
    #
    #
    # # counter = 0
    # for bbox_json in tqdm(bbox_list, desc="Processing bounding boxes"):
    #     tp = bbox_json.split(".")[0]
    #     ann = np.load(f"{ann_dir}/{tp}.npz")
    #     tp = int(tp[:13]) if len(tp) > 13 else int(tp)
    #
    #     bbox_data = json.load(open(f"{instance_bbox_dir}/{bbox_json}"))
    #     object_id_list = list(bbox_data.keys())
    #     # cluster object records
    #     for obj_id in object_id_list:
    #         find_collection(obj_id, bbox_data[obj_id], obj_record_dict)
    # print(f"processed {len(bbox_list)} frames of bbox")
    #
    # for abs_obj_id in obj_record_dict.keys():
    #     obj_frames = obj_record_dict[abs_obj_id]["frames"]
    #     prev_center = obj_frames[0]["center"]
    #     for i in range(len(obj_record_dict[abs_obj_id]["frames"])):
    #         # for obj_id in obj_record_dict[abs_obj_id]["recorded_id"]:
    #         one_frame = obj_frames[i]
    #         curr_center = one_frame["center"]
    #         distance = sum((a - b) ** 2 for a, b in zip(curr_center, prev_center)) ** 0.5
    #
    #         # Check if the distance exceeds the threshold
    #         moved = distance > 0.5
    #         obj_record_dict[abs_obj_id]["frames"][i]["has_moved"] = moved
    #         prev_center = curr_center
    #
    # obj_record_dict_clean = copy.deepcopy(obj_record_dict)
    # for abs_obj_id in obj_record_dict_clean.keys():
    #     obj_frames = obj_record_dict_clean[abs_obj_id]["frames"]
    #     del obj_record_dict_clean[abs_obj_id]["id"]
    #     del obj_record_dict_clean[abs_obj_id]["recorded_id"]
    #     for one_frame in obj_frames:
    #         del one_frame["id"]
    #         del one_frame["bbox3d"]
    # with open(f"{out_dir}/scene_graph.json", "w") as f:
    #     json.dump(obj_record_dict_clean, f, indent=4)
    # with open(f"{out_dir}/scene_graph_clean.json", "w") as f:
    #     json.dump(obj_record_dict_clean, f, indent=4)


if __name__ == "__main__":
    main()
