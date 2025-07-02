import os
import decord
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip
import glob
# 视频文件路径模式
#video_path_pattern = '../cogvideo_inference_output/long_video/10000_l6_first_i_con_125_cl_aesthetic_2_tiling_0.78_first_frame/inference_{}_video_0_The_f.mp4'
decord.bridge.set_bridge("torch")

def autoreg_video_save(base_path,suffix,num_videos,output_video_name="autoreg_video_1.mp4",key=0):
    
    
    video_path_pattern=os.path.join(base_path,suffix)
    # 视频数量
    # 存储所有帧的列表
    all_frames = []
    # 遍历所有视频文件
    for i in range(0, num_videos ):
        # 生成视频文件路径
        if key==0:
            video_path = video_path_pattern.format(i)
        else:
            video_path = video_path_pattern.format(i+1)
        # 读取视频
        vr = VideoReader(video_path, ctx=cpu(0))
        
        # 获取视频的所有帧
        frames = [frame.numpy() for frame in vr]
        print(len(frames[1:74]))
        
        # 如果是第一个视频，保留所有帧
    
        if i == 0:
            #continue
            all_frames.extend(frames[1:73])
        else:
            # 对于后续视频，去掉第一帧
            all_frames.extend(frames[8:73])

    # 将所有帧转换为视频
    output_video_path =os.path.join(base_path,output_video_name) 
    
    clip = ImageSequenceClip(all_frames, fps=30)  # 你可以根据需要调整fps
    print(len(all_frames))
    clip.write_videofile(output_video_path, codec='libx264')

    print(f"拼接视频已保存到 {output_video_path}")


if __name__=="__main__":
    #base_path="/home/cn/personilization/cogvideo_inference_output/long_video/binary_81cog15_output_1024/1000_l6_glm_first_i_con_12_cl_aesthetic_2_tiling_0.78/2"
    for i in range(2,6):
        base_path=f"/home/cn/personilization/cogvideo_inference_output/long_video/background_black_sketch_37/800_1360_seed42/l37_{i}"
        pattern="inference_*_video.mp4"
        #pattern="inference_*_video_0_Two_p.mp4"
        video_num=len(glob.glob(os.path.join(base_path,pattern)))
        print(video_num)
        autoreg_video_save(base_path,suffix="inference_{}_video.mp4",num_videos=video_num,
                        output_video_name="autoreg_video_1.mp4",key=1)