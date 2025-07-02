import cv2
import os

def save_frames_from_video(video_path, output_dir):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # 逐帧读取视频
        ret, frame = cap.read()
        if not ret:
            break

        # 保存当前帧为图像文件
        frame_filename = os.path.join(output_dir, f"{frame_count:03d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print(f"Saved {frame_count} frames to {output_dir}")

# 示例用法
video_path = "/home/cn/personilization/cogvideo_test_sample2/background_0_inc.mp4"
output_dir = "/home/cn/personilization/cogvideo_test_sample2/background_0"
save_frames_from_video(video_path, output_dir)
