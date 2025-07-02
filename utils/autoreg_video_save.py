import os
import decord
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip

# 视频文件路径模式
#video_path_pattern = '../cogvideo_inference_output/long_video/10000_l6_first_i_con_125_cl_aesthetic_2_tiling_0.78_first_frame/inference_{}_video_0_The_f.mp4'
video_path_pattern = "/home/cn/personilization/cogvideo_inference_output/long_video/binary_81cog15_output_1024/10000_l6_first_i_con_1230_cl_aesthetic_2_tiling_0.78/2/inference_{}_video_0_Two_p.mp4"

# 视频数量
num_videos = 5

# 存储所有帧的列表
all_frames = []

# 遍历所有视频文件
for i in range(0, num_videos ):
    # 生成视频文件路径
    video_path = video_path_pattern.format(i)
    
    # 读取视频
    vr = VideoReader(video_path, ctx=cpu(0))
    
    # 获取视频的所有帧
    frames = [frame.asnumpy() for frame in vr]
    print(len(frames[:74]))
    
    # 如果是第一个视频，保留所有帧
    if i == 0:
        #continue
        all_frames.extend(frames[:74])
    else:
        # 对于后续视频，去掉第一帧
        all_frames.extend(frames[8:74])

# 将所有帧转换为视频
output_video_path = "/home/cn/personilization/cogvideo_inference_output/long_video/binary_81cog15_output_1024/10000_l6_first_i_con_1230_cl_aesthetic_2_tiling_0.78/2/autoreg_video_1.mp4"

clip = ImageSequenceClip(all_frames, fps=30)  # 你可以根据需要调整fps
print(len(all_frames))
clip.write_videofile(output_video_path, codec='libx264')

print(f"拼接视频已保存到 {output_video_path}")
