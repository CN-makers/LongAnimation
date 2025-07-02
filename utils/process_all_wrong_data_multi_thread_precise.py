#import torch
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import PosixPath
try:
    import decord
except ImportError:
    raise ImportError(
        "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
    )
decord.bridge.set_bridge("torch")


def from_time_2_second(time_str):
    # 使用 strptime 解析时间字符串
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    # 计算总秒数
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    #print(total_seconds)
    return total_seconds

def calculate_frames(time_str, fps):
    """根据时间字符串和帧率计算帧数"""
    total_seconds = from_time_2_second(time_str)
    frames = int(total_seconds * fps ) 
    return frames

# 读取数据
df = pd.read_parquet("/home/cn/Datasets/SakugaDataset/parquet/train_aesthetic/sakugadataset_train_aesthetic.parquet")
df = df[['identifier', 'scene_start_time', 'scene_end_time', 'fps', "text_description", "aesthetic_score", "dynamic_score"]]
df = df.dropna(subset=['scene_start_time', 'scene_end_time', 'fps', "text_description", "aesthetic_score", "dynamic_score"])


# 计算start_frame
df['start_frame'] = df.apply(lambda row: calculate_frames(row['scene_start_time'], row['fps']), axis=1)


df['identifier_video'] = df['identifier'].apply(lambda x: int(x.split(':')[0]))

base_path = '/home/cn/Datasets/SakugaDataset/split/train_aesthetic_start_frame'
rows_to_delete = []

print(df.shape)

def file_exists(file_path):
    """检查文件是否存在"""
    return os.path.exists(file_path)

# 定义检查函数
def check_row(index, row):
    folder_path = os.path.join(base_path, str(row['identifier_video']))

    start_time=from_time_2_second(row['scene_start_time'])
    end_time=from_time_2_second(row['scene_end_time'])
    fps=row['fps']
    #计算总number的有可能有问题？再看一次，明天
    total_frame_num=(end_time-start_time)*fps
    
    #读取video，然后判断这个文件的真实帧数

    #print(int(start_time*fps))
    if total_frame_num<89:
        return index

    if not os.path.exists(folder_path):
        return index
    
    
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        if len(os.listdir(folder_path)) == 0:
            return index
    
    
    frames=row["start_frame"]
    video_name=row["identifier"].split(':')[0]
    #print(frames)
    data_path_1=f'{video_name}-Scene-{frames}.mp4'
    data_path_2=f'{video_name}-Scene-{frames+1}.mp4'
    data_path_3=f'{video_name}-Scene-{frames-1}.mp4'
    fd1=os.path.join(base_path,video_name,data_path_1)
    fd2=os.path.join(base_path,video_name,data_path_2)
    fd3=os.path.join(base_path,video_name,data_path_3)
    
    
    #判断是不是直接有，如果有的话就直接保留，如果没有的话，就看看有没有加一或者减一，可以自己先在这边过滤一下
    if not (file_exists(fd1) or file_exists(fd2) or file_exists(fd3) ):
        print(fd1)
        return index
    
    file_path=None
    if os.path.exists(fd1):
        file_path=fd1
    elif os.path.exists(fd2):
        file_path=fd2
    elif os.path.exists(fd3):
        file_path=fd3   
    video_reader = decord.VideoReader(uri=PosixPath(file_path).as_posix())
    video_num_frames = len(video_reader)
    if video_num_frames<89:
        print("video_num_frames",video_num_frames)
        return index
    #添加一个新的判断，判断start_frames是否在数据集中的地址，如果不在，就也返回index    
    
    return None

# 设置进度条
progress_dataset_bar = tqdm(total=df.shape[0], desc="Loading videos")

# 使用多线程执行检查
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    for index, row in df.iterrows():
        futures.append(executor.submit(check_row, index, row))

    # 收集结果
    for future in tqdm(futures, desc="Processing results"):
        result = future.result()
        if result is not None:
            rows_to_delete.append(result)
        progress_dataset_bar.update(1)

progress_dataset_bar.close()

# 删除满足条件的行
df.drop(rows_to_delete, inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape)

# 保存过滤后的数据
output_parquet_path ="/home/cn/Datasets/SakugaDataset/parquet/fliter_89_aesthetic_precise.parquet"
df.to_parquet(output_parquet_path, index=False)



#1054702 *8