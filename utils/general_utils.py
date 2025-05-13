import os
import numpy as np
import matplotlib.cm as cm
import cv2
import re
from tqdm import tqdm


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def obj_mask_viz(tp, full_img, labels, masks, bboxes, confidences, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    output_img = full_img.copy()
    n_objects = len(labels)

    # Use the 'tab20' colormap for up to 20 colors, or 'hsv' for more
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
    alpha = 0.5  # Transparency factor
    output_img = cv2.addWeighted(output_img, 1, mask_overlay, alpha, 0)

    # Save the output image
    output_path = f"{out_dir}/{tp}_masked.png"
    cv2.imwrite(output_path, output_img)

def create_video_from_images(image_folder, output_path, fps=30, frame_size=None):
    """
    Create MP4 video from a folder of images.

    Args:
        image_folder: Path to folder containing images
        output_path: Path where the video will be saved
        fps: Frames per second (default: 30)
        frame_size: Tuple (width, height) for frame size. If None, uses first image size
    """
    # Get list of image files
    images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    for file in os.listdir(image_folder):
        if file.lower().endswith(valid_extensions):
            images.append(file)

    # Sort images numerically (e.g., img1.png, img2.png, ..., img10.png)
    images.sort(key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else 0)

    if not images:
        print(f"No valid images found in {image_folder}")
        return False

    # Read the first image to get dimensions if frame_size not specified
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    if first_image is None:
        print(f"Error reading first image: {images[0]}")
        return False

    if frame_size is None:
        height, width = first_image.shape[:2]
        frame_size = (width, height)
    else:
        width, height = frame_size

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1' for H.264
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False

    # Write images to video
    for image_file in tqdm(images, desc="Creating video"):
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {image_file}, skipping...")
            continue

        # Resize image if necessary
        if (img.shape[1], img.shape[0]) != frame_size:
            img = cv2.resize(img, frame_size)

        video_writer.write(img)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to: {output_path}")
    return True


def downsample_frames(img_path_list, original_hz=10, target_hz=2):
    # Calculate the sampling interval
    interval = original_hz // target_hz

    # Return every 'interval'-th frame from the list
    downsampled_list = img_path_list[::interval]

    return downsampled_list

