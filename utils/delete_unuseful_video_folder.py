import os
from pathlib import Path

# 定义包含损坏文件路径的文本文件路径
error_file_path = '/home/cn/personilization/CogVideo15/unreadable_videos_81.txt'

# 读取文本文件中的所有损坏文件路径
with open(error_file_path, 'r') as file:
    lines = file.readlines()

# 提取每个文件路径的上一级目录
directories_to_delete = set()
for line in lines:
    line = line.strip()  # 去除行末的换行符
    if line:
        file_path = Path(line)
        parent_dir = file_path.parent
        directories_to_delete.add(parent_dir)

# 列出并删除这些上一级目录
for directory in directories_to_delete:
    try:
        # 列出要删除的目录
        print(f"Deleting directory: {directory}")
        # 删除目录及其内容
        #os.system(f"rm -rf {directory}")
    except Exception as e:
        print(f"Could not delete directory: {directory}. Error: {e}")

print("Deletion process completed.")
