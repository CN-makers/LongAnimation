from PIL import Image

# 打开 GIF 文件
gif_path = '/home/cn/Datasets/SakugaDataset/assets/gif/nobitanofriend-Scene-0021_1_fps14.gif'
gif = Image.open(gif_path)

# 获取 GIF 的第一帧
first_frame = gif.copy()

# 保存第一帧为单独的图像文件
first_frame_path = '/home/cn/personilization/CogVideo/test/first_frame.png'
first_frame.save(first_frame_path, format='PNG')

print(f"First frame saved to {first_frame_path}")
