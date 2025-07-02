import os

# 定义记录文件路径
file_list_path = '/home/cn/personilization/CogVideo15/unreadable_videos_81.txt'

# 检查记录文件是否存在
if not os.path.exists(file_list_path):
    print(f"Error: The file {file_list_path} does not exist.")
    exit(1)

# 读取文件路径列表
with open(file_list_path, 'r') as file:
    file_paths = file.readlines()

# 删除文件并记录删除结果
for file_path in file_paths:
    # 去掉路径字符串的换行符
    file_path = file_path.strip()
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("File deletion process completed.")
