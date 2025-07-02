import torch
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import shutil

def from_time_2_second(time_str):
    # 使用 strptime 解析时间字符串
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    # 计算总秒数
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    #print(total_seconds)
    return total_seconds


# 读取数据
df = pd.read_parquet("/home/cn/Datasets/SakugaDataset/parquet/test/sakugadataset_test.parquet")
df = df[['identifier', 'scene_start_time', 'scene_end_time', 'fps', "text_description", "aesthetic_score", "dynamic_score"]]
df = df.dropna(subset=['scene_start_time', 'scene_end_time', 'fps', "text_description", "aesthetic_score", "dynamic_score"])
df['identifier_video'] = df['identifier'].apply(lambda x: int(x.split(':')[0]))

base_path = '/private/task/cn/Datasets/SakugaDataset-main/download/split/test'
rows_to_delete = []

print(df.shape)

# 定义检查函数
def check_row(index, row):
    folder_path = os.path.join(base_path, str(row['identifier_video']))

    start_time=from_time_2_second(row['scene_start_time'])
    end_time=from_time_2_second(row['scene_end_time'])
    fps=row['fps']
    total_frame_num=(end_time-start_time)*fps
    if total_frame_num<300:
        return index

    # if not os.path.exists(folder_path):
    #     return index
    # if os.path.exists(folder_path) and os.path.isdir(folder_path):
    #     if len(os.listdir(folder_path)) == 0:
    #         return index
    return None

# 设置进度条
progress_dataset_bar = tqdm(total=df.shape[0], desc="Loading videos")

# 使用多线程执行检查
with ThreadPoolExecutor(max_workers=24) as executor:
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

aesthetic_median = df['aesthetic_score'].median()
dynamic_median = df['dynamic_score'].median()
print(aesthetic_median)
print(dynamic_median)

#aesthetic_median = df['aesthetic_score'].nlargest(len(df) // 3)

aesthetic_median = 0.8
# #dynamic_median = df['dynamic_score'].nlargest(len(df) // 3)
dynamic_median =0.4

print(f"Aesthetic Score Median: {aesthetic_median}")
print(f"Dynamic Score Median: {dynamic_median}")
filtered_df = df[(df['aesthetic_score'] >= aesthetic_median) & (df['dynamic_score'] >= dynamic_median)]

print(filtered_df.shape)

all_frame=0
for index, row in filtered_df.iterrows():
    start_time=from_time_2_second(row['scene_start_time'])
    end_time=from_time_2_second(row['scene_end_time'])
    fps=row['fps']
    total_frame_num=(end_time-start_time)*fps
    print(total_frame_num)
    all_frame+=total_frame_num
print(all_frame)


# 保存过滤后的数据
# output_parquet_path ="/private/task/cn/Datasets/SakugaDataset-main/download/parquet/fliter_196_test.parquet"
# df.to_parquet(output_parquet_path, index=False)




'''
#基础路径
base_path ='/private/task/cn/Datasets/SakugaDataset-main/download/split/test'
destination_path = '/private/task/cn/Datasets/SakugaDataset-main/download/test_dataset'

# 确保目标目录存在
os.makedirs(destination_path, exist_ok=True)

# 循环遍历过滤后的DataFrame
for index, row in filtered_df.iterrows():
    folder_path = os.path.join(base_path, str(row['identifier_video']))
    
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 构建目标路径
        target_path = os.path.join(destination_path, str(row['identifier_video']))
        if not os.path.exists(target_path):
            # 复制文件夹
            shutil.copytree(folder_path, target_path)
            print(f"Copied {folder_path} to {target_path}")
    else:
        print(f"Folder {folder_path} does not exist")




output_parquet_path ="/private/task/cn/Datasets/SakugaDataset-main/download/parquet/fliter_196_test_high_and.parquet"
filtered_df.to_parquet(output_parquet_path, index=False)


#1054702 *8
'''