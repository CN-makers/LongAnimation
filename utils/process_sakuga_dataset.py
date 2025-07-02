import torch
import pandas as pd
import os
from tqdm import tqdm 
df = pd.read_parquet("../../Datasets/SakugaDataset/parquet/train_aesthetic/sakugadataset_train_aesthetic.parquet")
df = df[['identifier', 'scene_start_time', 'scene_end_time', 'fps',"text_description","aesthetic_score","dynamic_score"]]
# drop rows with nan
df = df.dropna(subset=['scene_start_time', 'scene_end_time', 'fps',"text_description","aesthetic_score","dynamic_score"])
df['identifier_video'] = df['identifier'].apply(lambda x: int(x.split(':')[0]))

base_path = '/home/cn/Datasets/SakugaDataset/split/train_aesthetic'
rows_to_delete = []

print(df.shape)

# 遍历数据框的每一行
for index, row in df.iterrows():
    folder_path = os.path.join(base_path, str(row['identifier_video']))
    #print(str(row['identifier_video']))
    # 检查文件夹是否存在
    #print(folder_path)
    if not os.path.exists(folder_path):
        print(folder_path)
        rows_to_delete.append(index)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 检查文件夹中的文件数量
        if len(os.listdir(folder_path)) == 0:
            rows_to_delete.append(index)
            #print(index)

# 删除满足条件的行
df.drop(rows_to_delete, inplace=True)
# 重置索引
df.reset_index(drop=True, inplace=True)
print(df.shape)

output_parquet_path = '/home/cn/Datasets/SakugaDataset/parquet/fliter_aesthetic.parquet'
df.to_parquet(output_parquet_path, index=False)



#132337
# 132102  
# 删完无法读取的部分之后剩下了132067个

'''
print(df)

i=10
df.iloc[i]['identifier_video']


print(df.columns)
# 查看前几行数据
print(df.head())
print(df.iloc[1])
for index,row in df.iterrows():
    print(f"Row {index}: {row.to_dict()}")
    break
'''