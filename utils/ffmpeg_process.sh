ffmpeg -i video1.mp4 -vf "scale=1024:576" video1_576.mp4
ffmpeg -i video2.mp4 -vf "scale=-1:576" video2_576.mp4
ffmpeg -i video3.mp4 -vf "scale=-1:576" video3_576.mp4