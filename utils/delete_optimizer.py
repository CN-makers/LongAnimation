import os

# 指定顶层目录
top_dir = '/home/cn/personilization/cogvideo_output/cogvideox-15_sakuga_sketch_binary_81_1024_89'

# 遍历顶层目录及其子目录
for root, dirs, files in os.walk(top_dir):
    for file in files:
        if file == 'optimizer.bin':
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f'Removed: {file_path}')
