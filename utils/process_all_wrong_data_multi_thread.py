from pathlib import PosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed
import decord
from tqdm import tqdm

# 检查 decord 是否安装
try:
    import decord
except ImportError:
    raise ImportError(
        "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
    )

# 设置 decord 使用的桥接
decord.bridge.set_bridge("torch")

# 从文件中读取行
def read_lines_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                print(line.strip())  # 使用 strip() 去除行末的换行符
    except Exception as e:
        print(f"An error occurred: {e}")

# 将文件行读取到列表中
def read_lines_to_list(file_path):
    lines_list = []
    try:
        with open(file_path, 'r') as file:
            lines_list = [line.strip() for line in file]  # 使用列表推导式逐行读取并去除行末换行符
    except Exception as e:
        print(f"An error occurred: {e}")
    return lines_list

# 处理单个视频文件
def process_video(file, error_file_path):
    filename = PosixPath(file)
    try:
        video_reader = decord.VideoReader(uri=filename.as_posix())
    except Exception as e:
        with open(error_file_path, 'a') as f:
            f.write(f"{file}\n")
        print(f"Could not read video: {file}. Error: {e}")

# 使用示例
file_path = '/home/cn/Datasets/SakugaDataset/output_81.txt'
file_list = read_lines_to_list(file_path)
error_file_path = 'unreadable_videos_81.txt'

# 使用 ThreadPoolExecutor 实现多线程处理
with ThreadPoolExecutor(max_workers=16) as executor:  # 可以根据需要调整 max_workers 的数量
    futures = {executor.submit(process_video, file, error_file_path): file for file in file_list}
    for future in tqdm(as_completed(futures), total=len(file_list)):
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred: {e}")
