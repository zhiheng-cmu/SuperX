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

def trim_tp(tp):
    if len(str(tp)) > 11:
        return int(str(tp)[:11])
    else: return int(tp)


def main():
    # root_dir = "/media/zl3466/新加卷/dataset/SuperX"
    start_tp = 1744413306337674496
    end_tp = 1744413334057229568

    # data_dir = "/media/zl3466/新加卷/dataset/SuperX/set_0"
    # data_dir = "/Users/zhihengli/Downloads/SuperX/set_0"
    data_dir = "/home/zl3466/Documents/dataset/SuperX/tracking"
    out_dir = f"{data_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)

    img_dir = f"{data_dir}/image"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum"
    # ann_dir = f"{data_dir}/annotations"
    # instance_bbox_dir = f"{data_dir}/gates_mapping"

    bbox_list = os.listdir(instance_bbox_dir)
    bbox_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # ann_list = os.listdir(ann_dir)
    # ann_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    status_list = []
    obj_frame_dict = {}
    last_status_dict = {}
    for bbox_json in tqdm(bbox_list, desc="Processing bounding boxes"):
        bbox_data = json.load(open(f"{instance_bbox_dir}/{bbox_json}"))
        obj_list = list(bbox_data.keys())
        # tp = int(bbox_json.split(".")[0])
        # if tp < start_tp or tp > end_tp:
        #     print(tp, start_tp, end_tp)
        #     continue
        for obj_id in obj_list:
            obj_record = bbox_data[obj_id]
            if obj_record["status"] not in status_list:
                status_list.append(obj_record["status"])
            # a new object
            if obj_id not in obj_frame_dict.keys():
                last_status_dict[obj_id] = None
                if obj_record["status"] == "disappeared":
                    obj_frame_dict[obj_id] = [
                        {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "status": obj_record["status"]
                        }
                    ]
                else:
                    obj_frame_dict[obj_id] = [
                        {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "center": obj_record["center"],
                            "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                            "status": obj_record["status"]
                        }
                    ]
            # a seen object
            else:
                # if this is a change of status
                if obj_record["status"] != obj_frame_dict[obj_id][-1]["status"] and last_status_dict[obj_id] is not None:
                    obj_frame_dict[obj_id].append(last_status_dict[obj_id])
                    last_status_dict[obj_id] = None

                # handle different status
                if obj_record["status"] == "disappeared":
                    # do not record repeated disappeared status
                    if obj_frame_dict[obj_id][-1]["status"] == "disappeared":
                        continue
                    else:
                        if obj_id == "35":
                            print(f"{obj_frame_dict[obj_id][-1]['status']}, yesyes")
                        obj_frame_dict[obj_id].append(
                            {
                                "tp": trim_tp(obj_record["latest_stamp"]),
                                "id": obj_record["id"],
                                "label": obj_record["label"],
                                "status": obj_record["status"]
                            }
                        )
                elif obj_record["status"] == "persistent":
                    # do not record repeated persistent status. Save the last repeated persistent record in last_status_dict
                    if obj_frame_dict[obj_id][-1]["status"] == "persistent":
                        last_status_dict[obj_id] = {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "center": obj_record["center"],
                            "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                            "status": obj_record["status"]
                        }
                        continue
                    else:
                        obj_frame_dict[obj_id].append(
                            {
                                "tp": trim_tp(obj_record["latest_stamp"]),
                                "id": obj_record["id"],
                                "label": obj_record["label"],
                                "center": obj_record["center"],
                                "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                                "status": obj_record["status"]
                            }
                        )
                else:
                    obj_frame_dict[obj_id].append(
                        {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "center": obj_record["center"],
                            "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                            "status": obj_record["status"]
                        }
                    )
    # end of scene
    for obj_id in obj_frame_dict.keys():
        # if the object has not-yet-recorded repeated last frame
        if last_status_dict[obj_id] is not None and last_status_dict[obj_id]["tp"] != obj_frame_dict[obj_id][-1]["tp"]:
            obj_frame_dict[obj_id].append(last_status_dict[obj_id])
            last_status_dict[obj_id] = None


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
            # if "spatial_relations" in one_frame.keys():
            #     del one_frame["spatial_relations"]

    with open(f"{out_dir}/scene_graph_clean.json", "w") as f:
        json.dump(clean_frame_dict, f, indent=4)
    print(status_list)



if __name__ == "__main__":
    main()
