import cv2
import os
import re
from tqdm import tqdm
from utils.general_utils import *





def create_video_from_specific_images(image_paths, output_path, fps=10, frame_size=None):
    """
    Create MP4 video from a list of specific image paths.

    Args:
        image_paths: List of image file paths
        output_path: Path where the video will be saved
        fps: Frames per second (default: 30)
        frame_size: Tuple (width, height) for frame size. If None, uses first image size
    """
    if not image_paths:
        print("No image paths provided")
        return False

    # Read the first image to get dimensions if frame_size not specified
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"Error reading first image: {image_paths[0]}")
        return False

    if frame_size is None:
        height, width = first_image.shape[:2]
        frame_size = (width, height)
    else:
        width, height = frame_size

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False

    # Write images to video
    for img_path in tqdm(image_paths, desc="Creating video"):
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_path}, skipping...")
            continue

        # Resize image if necessary
        if (img.shape[1], img.shape[0]) != frame_size:
            img = cv2.resize(img, frame_size)

        video_writer.write(img)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to: {output_path}")
    return True


def obj_seg_id_viz(tp, full_img, labels, masks, bboxes, confidences, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    output_img = full_img.copy()
    n_objects = len(labels)
    colormap = cm.hsv

    # Generate colors and convert to BGR for OpenCV
    colors = []
    for i in range(n_objects):
        # Generate color from colormap (normalized to [0, 1])
        rgb = colormap(i / max(1, n_objects - 1))[:3]
        # Convert to BGR (OpenCV format) and scale to [0, 255]
        bgr = tuple(int(c * 255) for c in rgb[::-1])
        colors.append(bgr)

    # Create a mask overlay for all objects
    mask_overlay = np.zeros_like(full_img)

    # Project each mask onto the image
    for i in range(len(labels)):
        label = labels[i]
        mask = masks[i]
        bbox = bboxes[i]
        confidence = confidences[i]

        # Get color for this object
        color = colors[i]

        # Create colored mask
        colored_mask = np.zeros_like(full_img)
        colored_mask[mask > 0] = color

        # Add to the overlay
        mask_overlay = np.where(mask[..., np.newaxis] > 0, colored_mask, mask_overlay)

        # Draw bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

        # Add label text
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(output_img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Combine the original image with the mask overlay
    alpha = 0.4  # Transparency factor
    output_img = cv2.addWeighted(output_img, 1, mask_overlay, alpha, 0)

    # Save the output image
    output_path = f"{out_dir}/{tp}_masked.png"
    cv2.imwrite(output_path, output_img)
    

# Example usage
if __name__ == "__main__":
    # Example 3: Integrate with your code to create video from masked images
    # root_dir = "/home/zl3466/Documents/dataset/SuperX/set_0"
    # img_folder = f"{root_dir}/img_seg"

    root_dir = "/Users/zhihengli/Downloads/SuperX"
    data_dir = f"{root_dir}/set_0"
    seg_img_folder = f"{data_dir}/img_seg"

    out_dir = f"{root_dir}/outputs"
    os.makedirs(out_dir, exist_ok=True)
    output_video = f"{root_dir}/seg_video.mp4"

    # Get all masked images


    img_dir = f"{data_dir}/images"
    instance_bbox_dir = f"{data_dir}/json_serialization_mecanum_tiny"
    ann_dir = f"{data_dir}/annotations"

    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    bbox_list = os.listdir(instance_bbox_dir)
    bbox_list = [f for f in bbox_list if f.endswith('.json')]
    bbox_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
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
        obj_mask_viz(tp, full_img, labels, masks, bboxes, confidences, seg_img_folder)


    masked_images = []
    for file in os.listdir(seg_img_folder):
        if file.endswith('.png'):
            masked_images.append(os.path.join(seg_img_folder, file))

    # Sort them numerically
    masked_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # Create video
    create_video_from_specific_images(masked_images, output_video)
