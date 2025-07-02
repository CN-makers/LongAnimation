import os
import decord
from decord import VideoReader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 指定路径
root_path = '/home/cn/Datasets/SakugaDataset/split/train_aesthetic'
error_file = 'error_ratio.txt'

# 获取所有文件夹
folders = [os.path.join(root_path, folder) for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]

# 定义一个函数来处理每个文件夹
def process_folder(folder):
    try:
        # 获取文件夹中的任意一个视频文件
        video_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
        if not video_files:
            return None, None
        
        # 读取视频
        video_path = video_files[0]
        vr = VideoReader(video_path)
        
        # 获取视频帧的宽高
        frame = vr[0]
        height, width = frame.shape[:2]
        ratio = width / height
        
        # 返回结果
        if ratio < 1:
            return folder, ratio
        else:
            return None, None
    except Exception as e:
        print(f"Error processing folder {folder}: {e}")
        return None, None

# 使用多线程处理
results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_folder, folder) for folder in folders]
    for future in tqdm(futures, desc="Processing folders"):
        folder, ratio = future.result()
        if folder is not None:
            results.append((folder, ratio))

# 打印长宽比小于1的位置和长宽比，并写入文件
with open(error_file, 'w') as f:
    for folder, ratio in results:
        print(f"Folder: {folder}, Ratio: {ratio}")
        f.write(f"{folder}\n")
