import os

# 文件路径
loss_file_path = '/home/cn/personilization/CogVideo15/abnormal_loss_base.txt'

# 读取文件内容
with open(loss_file_path, 'r') as file:
    lines = file.readlines()

# 遍历每一行
for line in lines:
    # 提取文件路径和损失值
    parts = line.strip().split(':')
    file_path = parts[0].strip("[]' ")
    loss_value = float(parts[1].strip())

    # 检查损失值是否大于0.6
    if loss_value > 0.6:
        # 删除文件
        if os.path.exists(file_path):
            print(file_path)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"File not found: {file_path}")

print("Completed processing.")
