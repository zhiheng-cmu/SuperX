import os
import json
from tqdm import tqdm
import numpy as np

'''
{
    timestamp,
    curr_id,
    past_id: [id, id, id, id],
    label, # detailed if possible, e.g. red cup
    center,
    temporal_relationship: [Disappeared / Newly Appeared / Persistent / Highly Dynamic] # select one of them
    spatial_relationship: {
        in: [],
        on: [obj],
        next_to: [obj1, obj2],
        under: []
    }
}
'''

def main():
    root_dir = "/Users/zhihengli/Downloads/SuperX"
    data_dir = f"{root_dir}/mini"
    out_dir = f"{root_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)

    img_dir = f"{data_dir}/images"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum"
    ann_dir = f"{data_dir}/annotations"

    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    bbox_list = os.listdir(instance_bbox_dir)
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
        ann = np.load(f"{ann_dir}/{tp}.npz")
        full_img = cv2.imread(f"{img_dir}/{tp}.png")

        labels = ann["labels"]
        bboxes = ann["bboxes"]
        masks = ann["masks"]
        confidences = ann["confidences"]

        # print(labels)
        # print(bboxes)
        # print(masks)
        # print(confidences)
        obj_mask_viz(tp, full_img, labels, masks, bboxes, confidences, f"{data_dir}/img_seg")
