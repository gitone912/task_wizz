
# !git clone https://github.com/WongKinYiu/yolov7
# %cd yolov7


# !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

# !python detect.py --weights ./yolov7-tiny.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

# from PIL import Image
# Image.open('/Users/pranaymishra/Desktop/employee_monitoring/test/yolov7/runs/detect/exp/horses.jpg')

import cv2
import os

# Step 1: Extract frames from the video
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(5)  # Frame rate of the video
    interval = int(frame_rate * 30)  # Extract every half minute frame
    count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_name, frame)

        count += 1

    cap.release()

# Step 2: Detect objects in each frame
def detect_objects(weights_path, conf_threshold, img_size, source_folder, output_folder):
    detect_command = f"python yolov7/detect.py --weights {weights_path} --conf {conf_threshold} --img-size {img_size} --source {source_folder}"

    os.system(detect_command)

# Step 3: Create video from detected frames
def create_video(frames_folder, output_video_path):
    frames = [f for f in os.listdir(frames_folder) if f.endswith(".jpg")]
    frames.sort()

    frame = cv2.imread(os.path.join(frames_folder, frames[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    for frame_name in frames:
        frame_path = os.path.join(frames_folder, frame_name)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()

def main():
    video_path = "/Users/pranaymishra/Desktop/employee_monitoring/your_video.mp4"
    output_folder = "frames"
    weights_path = "./yolov7-tiny.pt"
    conf_threshold = 0.25
    img_size = 640
    source_folder = "inference/images"
    output_video_path = "output_video.mp4"

    extract_frames(video_path, output_folder)
    detect_objects(weights_path, conf_threshold, img_size, source_folder, output_folder)
    create_video(output_folder, output_video_path)


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(5)  # Frame rate of the video
    interval = int(frame_rate * 10)  # Extract every half minute frame
    count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_name, frame)

        count += 1

    cap.release()

video_path = "/Users/pranaymishra/Desktop/employee_monitoring/your_video.mp4"
output_folder = "frames"
    

extract_frames(video_path, output_folder)



