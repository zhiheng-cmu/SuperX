import json
import os

import numpy as np
import cv2

# Dummy camera intrinsics (fx, fy, cx, cy)
K = np.array([
    [525.0, 0.0, 319.5],
    [0.0, 525.0, 239.5],
    [0.0, 0.0, 1.0]
])

def get_3d_bbox_corners(center, extent):
    dx, dy, dz = extent[0] / 2, extent[1] / 2, extent[2] / 2
    cx, cy, cz = center

    corners = np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ])
    return corners

def project_to_image(points_3d, K):
    points_2d = K @ points_3d.T
    points_2d = points_2d[:2] / points_2d[2]
    return points_2d.T.astype(int)

def draw_bbox(img, points_2d, color=(0, 255, 0)):
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in lines:
        pt1 = tuple(points_2d[i])
        pt2 = tuple(points_2d[j])
        cv2.line(img, pt1, pt2, color, 2)



data_dir = "/home/zl3466/Documents/dataset/SuperX/mini"
img_dir = f"{data_dir}/images/1736999024763313408.png"
instance_bbox = f"{data_dir}/json_serialization_mecanum/1736999033954955776.json"
room_bbox = f"{data_dir}/cic_room_bbox_json/1736999024763313408.json"

# Load your JSON file
with open(instance_bbox) as f:
    data = json.load(f)

# Load the image
image = cv2.imread(img_dir)
print(img_dir)
print(os.path.exists(img_dir))

# Parse and draw each bounding box
for obj_key in data.keys():  # update this based on actual structure
    obj = data[obj_key]
    label = obj['label']
    bbox = obj['bbox3d']

    center = [bbox['center'][0], bbox['center'][1], bbox['center'][2]]
    extent = [bbox['extent'][0], bbox['extent'][1], bbox['extent'][2]]

    corners_3d = get_3d_bbox_corners(center, extent)
    points_2d = project_to_image(corners_3d, K)
    draw_bbox(image, points_2d)

    # Optional: Put label
    cv2.putText(image, label, tuple(points_2d[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

cv2.imshow('3D Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
