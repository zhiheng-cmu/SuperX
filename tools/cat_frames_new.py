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
    if len(str(tp)) > 13:
        return int(str(tp)[:13])
    else: return int(tp)


def round_center(center, num_digit=3):
    return [round(num, num_digit) for num in center]


def merge_same_obj(obj_frame_dict, threshold=0.05):
    result = copy.deepcopy(obj_frame_dict)
    for obj_id in tqdm(obj_frame_dict.keys(), desc="Merging objects"):
        if obj_id not in result.keys():
            continue
        obj1 = obj_frame_dict[obj_id][0]

        # for obj_id_2 in tqdm(obj_list, desc=f"walking through all objects with respect to {obj_id}"):
        for obj_id_2 in obj_frame_dict.keys():
            obj2 = obj_frame_dict[obj_id_2][0]

            # filter out faulty object that starts with a "disappeared" status
            if "center" not in obj1.keys():
                if obj_id in result.keys():
                    del result[obj_id]
                continue
            if "center" not in obj2.keys():
                if obj_id_2 in result.keys():
                    del result[obj_id_2]
                continue
            # only merge objects with the same label; only merge objects that have not been merged before
            if obj_id_2 == obj_id or obj1["label"] != obj2["label"]:
                continue
            else:
                center1 = obj1["center"]
                center2 = obj2["center"]

                # if the two objects have the same label and are extremely close to each other,
                # we deem them as the same obj and merge their frames
                if abs(center1[0] - center2[0]) < threshold and abs(center1[1] - center2[1]) < threshold and abs(center1[2] - center2[2]) < threshold:
                    print(f"merging {obj1['label']} {obj_id_2} into  {obj_id}")
                    # merge the obj2 into obj1
                    obj_frame_dict[obj_id_2][0]["status"] = "persistent"
                    obj_frame_dict[obj_id] += obj_frame_dict[obj_id_2]
                    obj_frame_dict[obj_id] = sorted(obj_frame_dict[obj_id], key=lambda x: x["tp"])

                    # update the id recorded in other objects' spatial relations
                    # substitute all obj_id_2 with obj_id
                    for obj_id_tmp in obj_frame_dict.keys():
                        obj_tmp_frames = obj_frame_dict[obj_id_tmp]
                        for frame in obj_tmp_frames:
                            if "spatial_relations" in frame.keys():
                                for relation in frame["spatial_relations"].keys():
                                    if obj_id_2 in frame["spatial_relations"][relation]:
                                        # substitute the obj_id_2 in frame["spatial_relations"][relation] with obj_id_1
                                        frame["spatial_relations"][relation] = [obj_id if obj == obj_id_2 else obj for obj
                                                                                in frame["spatial_relations"][relation]]
                                        # remove duplicate
                                        for relation_id in frame["spatial_relations"][relation]:
                                            if relation_id == obj_id:
                                                frame["spatial_relations"][relation].remove(obj_id)

                                        frame["spatial_relations"][relation] = list(dict.fromkeys(frame["spatial_relations"][relation]))
                    if obj_id_2 in result.keys():
                        del result[obj_id_2]
    # sort the frames by tp
    for obj_id in obj_frame_dict.keys():
        frame_list = obj_frame_dict[obj_id]
        obj_frame_dict[obj_id] = sorted(frame_list, key=lambda x: x["tp"])

    for obj_id in result.keys():
        result[obj_id] = obj_frame_dict[obj_id]

    print(f"before merge: {len(obj_frame_dict.keys())} objects, after merge: {len(result.keys())} objects")
    return result


BIG_OBJECTS = [
    'sofa', 'table', 'cabinet', 'refrigerator', 'chair', 'screen', 'painting', 'human', 'plant'
]

SMALL_OBJECTS = [
    'books', 'bottle', 'cup', 'bag'
]

def check_object_relations(obj1, obj2):
        """
            Check the spatial relations between two objects. Obj1 w.r.t. obj2
        """
        # check if two objects are adjacent
        centroid1 = np.array(obj1["center"])
        centroid2 = np.array(obj2["center"])
        if centroid1 is None or centroid2 is None:
            return None
        else:
            # Assume gravity is in the z direction
            vec = centroid2 - centroid1
            dist = np.linalg.norm(vec)
            if obj1.get_dominant_label() in BIG_OBJECTS and obj2.get_dominant_label() in BIG_OBJECTS:
                if dist < 0.5:
                    if np.linalg.norm(vec[:2]) < 0.5 and vec[2] > 0.0:
                        return "under"
                    elif np.linalg.norm(vec[:2]) < 0.5 and vec[2] < 0.0:
                        return "above"
                elif dist < 1.0:
                    return "beside"
                else:
                    return None
            elif obj1.get_dominant_label() in SMALL_OBJECTS and obj2.get_dominant_label() in SMALL_OBJECTS:
                if dist < 0.5:
                    return "beside"
                else:
                    return None
            elif obj1.get_dominant_label() in SMALL_OBJECTS and obj2.get_dominant_label() in BIG_OBJECTS:
                if dist < 1.0:
                    if np.linalg.norm(vec[:2]) < 0.5 and vec[2] > 0.0:
                        return "under"
                    elif np.linalg.norm(vec[:2]) < 0.5 and vec[2] < 0.0:
                        return "on"
            elif obj1.get_dominant_label() in BIG_OBJECTS and obj2.get_dominant_label() in SMALL_OBJECTS:
                if dist < 1.0:
                    if np.linalg.norm(vec[:2]) < 0.5 and vec[2] > 0.0:
                        return "above"
                    elif np.linalg.norm(vec[:2]) < 0.5 and vec[2] < 0.0:
                        return "on"
            else:
                return None





def main():
    # root_dir = "/media/zl3466/新加卷/dataset/SuperX"
    start_tp = 1744413306337674496
    end_tp = 1744413334057229568

    # data_dir = "/media/zl3466/新加卷/dataset/SuperX/set_0"
    data_dir = "/Users/zhihengli/Downloads/SuperX/tracking"
    # data_dir = "/home/zl3466/Documents/dataset/SuperX/tracking"
    out_dir = f"{data_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)
    split = "bag"

    img_dir = f"{data_dir}/image"
    # instance_bbox_dir = f"{data_dir}/json_serialization_mecanum"
    # ann_dir = f"{data_dir}/annotations"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum_{split}"

    bbox_list = os.listdir(instance_bbox_dir)
    bbox_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # ann_list = os.listdir(ann_dir)
    # ann_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    status_list = []
    obj_frame_dict = {}
    last_status_dict = {}
    last_dynamic_dict = {}
    for bbox_json in tqdm(bbox_list, desc="Processing bounding boxes"):
        bbox_data = json.load(open(f"{instance_bbox_dir}/{bbox_json}"))
        obj_list = list(bbox_data.keys())
        tp = int(bbox_json.split(".")[0])
        # if tp < start_tp or tp > end_tp:
        #     print(tp, start_tp, end_tp)
        #     continue
        for obj_id in obj_list:
            # if obj_id == "77" or obj_id == "2720":
            #     print(bbox_data[obj_id])
            obj_record = bbox_data[obj_id]
            if obj_record["status"] not in status_list:
                status_list.append(obj_record["status"])
            # a new object
            if obj_id not in obj_frame_dict.keys():
                last_status_dict[obj_id] = None
                last_dynamic_dict[obj_id] = None
                if obj_record["status"] == "disappeared":
                    obj_frame_dict[obj_id] = [
                        {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "status": obj_record["status"]
                        }
                    ]
                elif obj_record["status"] == "dynamic":
                    obj_frame_dict[obj_id] = [
                        {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "center": round_center(obj_record["center"]),
                            "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                            "status": obj_record["status"]
                        }
                    ]
                    last_dynamic_dict[obj_id] ={
                        "tp": trim_tp(obj_record["latest_stamp"]),
                        "id": obj_record["id"],
                        "label": obj_record["label"],
                        "center": round_center(obj_record["center"]),
                        "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                        "status": obj_record["status"]
                    }
                else:
                    obj_frame_dict[obj_id] = [
                        {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "center": round_center(obj_record["center"]),
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
                # disappeared
                if obj_record["status"] == "disappeared":
                    # do not record repeated disappeared status
                    if obj_frame_dict[obj_id][-1]["status"] == "disappeared":
                        continue
                    else:
                        # if obj_id == "665":
                        #     print(f"{obj_frame_dict[obj_id][-1]['tp']} {obj_record['latest_stamp']} {tp} {obj_frame_dict[obj_id][-1]['status']}, yesyes")
                        obj_frame_dict[obj_id].append(
                            {
                                "tp": trim_tp(tp),
                                "id": obj_record["id"],
                                "label": obj_record["label"],
                                "status": obj_record["status"]
                            }
                        )
                # persistent
                elif obj_record["status"] == "persistent":
                    # do not record repeated persistent status. Save the last repeated persistent record in last_status_dict
                    if obj_frame_dict[obj_id][-1]["status"] == "persistent":
                        last_status_dict[obj_id] = {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "center": round_center(obj_record["center"]),
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
                                "center": round_center(obj_record["center"]),
                                "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                                "status": obj_record["status"]
                            }
                        )
                # dynamic
                elif obj_record["status"] == "dynamic":
                    # do not record repeated persistent status. Save the last repeated persistent record in last_status_dict
                    if obj_frame_dict[obj_id][-1]["status"] == "dynamic":
                        # record consecutive dynamic at 1hz
                        if abs(last_dynamic_dict[obj_id]["tp"] - trim_tp(obj_record["latest_stamp"])) > 1000:
                            obj_frame_dict[obj_id].append(
                                {
                                    "tp": trim_tp(obj_record["latest_stamp"]),
                                    "id": obj_record["id"],
                                    "label": obj_record["label"],
                                    "center": round_center(obj_record["center"]),
                                    "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                                    "status": obj_record["status"]
                                }
                            )
                            last_dynamic_dict[obj_id] = {
                                "tp": trim_tp(obj_record["latest_stamp"]),
                                "id": obj_record["id"],
                                "label": obj_record["label"],
                                "center": round_center(obj_record["center"]),
                                "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                                "status": obj_record["status"]
                            }
                        else:
                            continue
                    else:
                        obj_frame_dict[obj_id].append(
                            {
                                "tp": trim_tp(obj_record["latest_stamp"]),
                                "id": obj_record["id"],
                                "label": obj_record["label"],
                                "center": round_center(obj_record["center"]),
                                "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                                "status": obj_record["status"]
                            }
                        )
                        last_dynamic_dict[obj_id] = {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "center": round_center(obj_record["center"]),
                            "spatial_relations": clean_spatial_relations(obj_record["spatial_relations"]),
                            "status": obj_record["status"]
                        }
                else:
                    obj_frame_dict[obj_id].append(
                        {
                            "tp": trim_tp(obj_record["latest_stamp"]),
                            "id": obj_record["id"],
                            "label": obj_record["label"],
                            "center": round_center(obj_record["center"]),
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

    clean_frame_dict = {key: clean_frame_dict[key] for key in sorted(clean_frame_dict.keys(), key=int)}
    with open(f"{out_dir}/scene_graph_clean_no_merge.json", "w") as f:
        json.dump(clean_frame_dict, f, indent=4)
    clean_frame_dict = merge_same_obj(clean_frame_dict, threshold=0.1)
    clean_frame_dict = {key: clean_frame_dict[key] for key in sorted(clean_frame_dict.keys(), key=int)}
    print(f"result saved to {out_dir}/scene_graph_clean_{split}.json")
    with open(f"{out_dir}/scene_graph_clean_{split}.json", "w") as f:
        json.dump(clean_frame_dict, f, indent=4)
    print(status_list)



if __name__ == "__main__":
    main()
