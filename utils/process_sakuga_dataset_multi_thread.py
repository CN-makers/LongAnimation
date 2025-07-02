import torch
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


def from_time_2_second(time_str):
    # 使用 strptime 解析时间字符串
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    # 计算总秒数
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    #print(total_seconds)
    return total_seconds


# 读取数据
df = pd.read_parquet("//home/cn/Datasets/SakugaDataset/parquet/fliter_49_aesthetic.parquet")
df = df[['identifier', 'scene_start_time', 'scene_end_time', 'fps', "text_description", "aesthetic_score", "dynamic_score"]]
df = df.dropna(subset=['scene_start_time', 'scene_end_time', 'fps', "text_description", "aesthetic_score", "dynamic_score"])
df['identifier_video'] = df['identifier'].apply(lambda x: int(x.split(':')[0]))

base_path = '/home/cn/Datasets/SakugaDataset/split/train_aesthetic'
rows_to_delete = []

print(df.shape)

# 定义检查函数
def check_row(index, row):
    folder_path = os.path.join(base_path, str(row['identifier_video']))

    start_time=from_time_2_second(row['scene_start_time'])
    end_time=from_time_2_second(row['scene_end_time'])
    fps=row['fps']
    total_frame_num=(end_time-start_time)*fps
    if total_frame_num<81:
        return index

    if not os.path.exists(folder_path):
        return index
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        if len(os.listdir(folder_path)) == 0:
            return index
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
output_parquet_path ="/home/cn/Datasets/SakugaDataset/parquet/fliter_train_81_2.parquet"
df.to_parquet(output_parquet_path, index=False)



#1054702 *8