import rosbag
import cv2
from cv_bridge import CvBridge


# Path to your bag file and topic
# data_dir = "/home/zl3466/Documents/dataset/SuperX/mini"
data_dir = "/Users/zhihengli/Downloads/SuperX"
img_bag_dir = f"{data_dir}/camera_images00.bag"

image_topic = '/camera/image'  # or your specific topic

# Initialize CvBridge
bridge = CvBridge()

# Open the bag file
bag = rosbag.Bag(img_bag_dir, 'r')

save_frames = True
save_path = f"{data_dir}/images"
if save_frames:
    import os
    os.makedirs(save_path, exist_ok=True)

# Read messages from the bag
for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[image_topic])):
    try:
        # Convert the ROS Image message to a CV2 image (OpenCV)
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Display (optional)
        cv2.imshow("RGB Image", cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save frame as image file
        if save_frames:
            cv2.imwrite(f"{save_path}/{t}.png", cv_image)

    except Exception as e:
        print(f"Error converting image {i}: {e}")

bag.close()
cv2.destroyAllWindows()
