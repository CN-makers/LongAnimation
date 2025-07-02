import cv2
import os
from diffusers.utils import export_to_video
from PIL import Image

# 设置图片文件夹路径和输出视频文件路径

#image_folder = 'test/sample_4'
image_folder = '/home/cn/personilization/cogvideo_test_sample2/background_0_sketch'
output_video_path = '/home/cn/personilization/cogvideo_test_sample2'

# 获取文件夹中的所有图片文件，并按文件名排序
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images=sorted(images,key=lambda x : int(x.split("_")[0]))
#images=sorted(images,key=lambda x : int(x.split(".")[0]))

# 只取前49帧
#start_frame=39
#start_frame=24
#start_frame=15 #sample4
#start_frame=15
#images = images[start_frame:start_frame+881]
print(images)


# 获取第一张图片的尺寸
first_image_path = os.path.join(image_folder, images[0])

first_image_path = os.path.join(image_folder, images[0])
first_image = Image.open(first_image_path)
width, height = first_image.size

# 计算新的尺寸，使其为16的倍数
new_width = (width + 15) // 16 * 16
new_height = (height + 15) // 16 * 16


frames = []
for image in images:
    img_path = os.path.join(image_folder, image)
    img = Image.open(img_path)
    #print(img.mode)
    # 调整图像尺寸
    resized_img = img.resize((new_width, new_height), Image.BILINEAR)
    frames.append(resized_img)

final_save_path=os.path.join(output_video_path, "background_0_sketch.mp4")
export_to_video(frames, final_save_path, fps=16)



