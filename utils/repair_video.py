from PIL import Image

def repair_gif(input_path, output_path):
    with Image.open(input_path) as img:
        frames = []
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0)


from moviepy.editor import VideoFileClip

def gif_to_mp4(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec='libx264')

input_path = '/home/cn/personilization/CogVideo/test/simple_3.gif'
output_path = '/home/cn/personilization/CogVideo/test/simple_3.mp4'
gif_to_mp4(input_path, output_path)



#repair_gif(input_path, output_path)
