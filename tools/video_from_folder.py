import cv2
import os
import re
from tqdm import tqdm


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


# Example usage
if __name__ == "__main__":
    # Example 3: Integrate with your code to create video from masked images
    root_dir = "/home/zl3466/Documents/dataset/SuperX/mini"
    img_folder = f"{root_dir}/img_seg"
    output_video = f"{root_dir}/seg_video.mp4"

    # Get all masked images
    masked_images = []
    for file in os.listdir(img_folder):
        if file.endswith('.png'):
            masked_images.append(os.path.join(img_folder, file))

    # Sort them numerically
    masked_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # Create video
    create_video_from_specific_images(masked_images, output_video)
