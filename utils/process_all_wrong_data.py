from pathlib import PosixPath
try:
    import decord
except ImportError:
    raise ImportError(
        "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
    )

from tqdm import tqdm
decord.bridge.set_bridge("torch")

def read_lines_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                print(line.strip())  # 使用 strip() 去除行末的换行符
                
                # filename=PosixPath(line)
                # video_reader = decord.VideoReader(uri=filename.as_posix())
    except Exception as e:
        print(f"An error occurred: {e}")

def read_lines_to_list(file_path):
    lines_list = []
    try:
        with open(file_path, 'r') as file:
            lines_list = [line.strip() for line in file]  # 使用列表推导式逐行读取并去除行末换行符
    except Exception as e:
        print(f"An error occurred: {e}")
    return lines_list

# 使用示例
file_path = '/home/cn/Datasets/SakugaDataset/output_81.txt'
file_list=read_lines_to_list(file_path)
error_file_path = 'unreadable_videos_81.txt'

for file in tqdm(file_list):
    filename=PosixPath(file)
    try:
        video_reader = decord.VideoReader(uri=filename.as_posix())
    except Exception as e:
        with open(error_file_path, 'a') as f:
            f.write(f"{file}\n")
        print(f"Could not read video: {file}. Error: {e}")