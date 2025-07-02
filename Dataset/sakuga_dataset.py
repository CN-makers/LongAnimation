
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
import numpy as np
import accelerate
import torch
import pandas as pd
from pathlib import PosixPath
import os 
from datetime import datetime

try:
    import decord
except ImportError:
    raise ImportError(
        "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
    )
decord.bridge.set_bridge("torch")


class Sakuga_Dataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        sketch_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        height: int = 480,
        width: int = 720,
        video_reshape_mode: str = "center",
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
        data_information:  Optional[str] = None,
        stage: Optional[str] = "1",
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.sketch_data_root = Path(sketch_data_root) if sketch_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.video_reshape_mode = video_reshape_mode
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""
        self.stage=stage
        
        '''
        if dataset_name is not None:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_hub()
        else:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()
        '''
        
        self.data_information=pd.read_parquet(data_information)
        self.num_instance_videos = self.data_information.shape[0]


        #self.detector = LineartDetector('cpu')
        #TODO: here just point the cuda maybe have some problem

        #we put the preprocess_data() in the get_item function
        #self.instance_videos = self._preprocess_data()
        #here, how to make it in the get_item?

    def __len__(self):
        return self.num_instance_videos
    
    def encode_video(self, video,vae,device):
        

        #vae,device
        video = video.to(device, dtype=vae.dtype).unsqueeze(0)
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        image = video[:, :, :1].clone()

        latent_dist = vae.encode(video).latent_dist

        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=image.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=image.dtype)
        noisy_image = torch.randn_like(image) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = vae.encode(noisy_image).latent_dist

        return latent_dist, image_latent_dist

    def read_video(self,video_path):
        filename=PosixPath(video_path)

        #this part have some wrong things
        try:
            video_reader = decord.VideoReader(uri=filename.as_posix())
            video_num_frames = len(video_reader)

            #需不需要这里强制一下从第10帧开始？
            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
            else:
                #this has problem
                #indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
                indices=list(range(start_frame,self.max_num_frames))
                frames = video_reader.get_batch(indices)

            # Ensure that we don't go over the limit
            frames = frames[: self.max_num_frames]
            selected_num_frames = frames.shape[0]

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0

            # Training transforms
            
            frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
            #print("frame",frames.shape)
            frames = self._resize_for_rectangle_crop(frames)
            final_frames = frames.contiguous()
            if final_frames.dim()==3:
                final_frames=final_frames.unsqueeze(0)
            return final_frames
        except:
            return None
        
        
    
    def __getitem__(self, index):
        
        #output_video=self.encode_video(video,vae,device)
        #_encode_instance_video=self.encode_video(self.instance_prompts[index],device=)
        
        #处理selfinstance_videos
        
        folder_path=os.path.join(self.instance_data_root, str(self.data_information.iloc[index]['identifier_video']))
        
        #sketch_path=os.path.join(self.sketch_data_root, str(self.data_information.iloc[index]['identifier_video']))
        #注意这里的sketch存的是[0,255]的信息，不需要1-了，但是之后可能还是要变成2值的操作，得看一下如何2值化
        
        
        #这里寻找id号的过程有点问题，不太对
        #这个identifier是不是就是对应的片段名称？是的，identifier后面的对应的就是sence_number
        #indices = self.data_information.index[self.data_information['identifier_video'] == self.data_information.iloc[index]['identifier_video']].tolist()
        
        # video_name=self.data_information.iloc[index]['identifier_video']
        # sence_number=self.data_information.iloc[index]["identifier"].split(":")[1]
        # data_name=f"{video_name}-Scene-{int(sence_number):03d}.mp4"

        
        frames=self.data_information.iloc[index]["start_frame"]
        video_name=self.data_information.iloc[index]["identifier"].split(':')[0]
        #print(frames)
        
        data_path_1=f'{video_name}-Scene-{frames}.mp4'
        data_path_2=f'{video_name}-Scene-{frames+1}.mp4'
        data_path_3=f'{video_name}-Scene-{frames-1}.mp4'
        fd1=os.path.join(folder_path,data_path_1)
        #sketch_fd1=os.path.join(sketch_path,data_path_1)
        fd2=os.path.join(folder_path,data_path_2)
        #sketch_fd2=os.path.join(sketch_path,data_path_2)
        fd3=os.path.join(folder_path,data_path_3)
        #sketch_fd3=os.path.join(sketch_path,data_path_2)
        
        #print(fd1)
        
        if os.path.exists(fd1):
            file_path=fd1
        elif os.path.exists(fd2):
            file_path=fd2
        elif os.path.exists(fd3):
            file_path=fd3   
        

        
        prompt=self.data_information.iloc[index]["text_description"]
                
        final_frames=self.read_video(PosixPath(file_path))
        #final_sketch_frames=self.read_video(PosixPath(sketch_file_path))
        final_sketch_frames=None
        
        
        
        instance_prompt =  prompt + self.id_token
        
        return {
            "instance_prompt": instance_prompt,
            "instance_video": final_frames,
            "file_path":file_path,
            "sketch_video": final_sketch_frames,
            "instance_image": None,
            #"instance_sketch": final_sketch,
        }

    def _load_dataset_from_hub(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                "local folder containing images only, specify --instance_data_root instead."
            )

        # Downloading and loading a dataset from the hub. See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            cache_dir=self.cache_dir,
        )
        column_names = dataset["train"].column_names

        if self.video_column is None:
            video_column = column_names[0]
            #logger.info(f"`video_column` defaulting to {video_column}")
            print(f"`video_column` defaulting to {video_column}")
        else:
            video_column = self.video_column
            if video_column not in column_names:
                raise ValueError(
                    f"`--video_column` value '{video_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.caption_column is None:
            caption_column = column_names[1]
            #logger.info(f"`caption_column` defaulting to {caption_column}")
            print(f"`caption_column` defaulting to {caption_column}")
        else:
            caption_column = self.caption_column
            if self.caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        instance_prompts = dataset["train"][caption_column]
        instance_videos = [Path(self.instance_data_root, filepath) for filepath in dataset["train"][video_column]]

        return instance_prompts, instance_videos

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        video_path = self.instance_data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            instance_videos = [
                self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
            ]

        if any(not path.is_file() for path in instance_videos):
            raise ValueError(
                "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return instance_prompts, instance_videos

    def _resize_for_rectangle_crop(self, arr):
        image_size = self.height, self.width
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr


    # here process the all data, we should make these processed in the get_item or other position

    def _preprocess_data(self):


        decord.bridge.set_bridge("torch")

        progress_dataset_bar = tqdm(
            range(0, len(self.instance_video_paths)),
            desc="Loading progress resize and crop videos",
        )

        videos = []

        for filename in self.instance_video_paths:
            video_reader = decord.VideoReader(uri=filename.as_posix())
            video_num_frames = len(video_reader)

            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
            else:
                indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
                frames = video_reader.get_batch(indices)

            # Ensure that we don't go over the limit
            frames = frames[: self.max_num_frames]
            selected_num_frames = frames.shape[0]

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0

            # Training transforms
            
            frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
            progress_dataset_bar.set_description(
                f"Loading progress Resizing video from {frames.shape[2]}x{frames.shape[3]} to {self.height}x{self.width}"
            )
            frames = self._resize_for_rectangle_crop(frames)  #here the tensor should be processed to right size
            
            frames = (frames - 127.5) / 127.5
            videos.append(frames.contiguous())  # [F, C, H, W]
            progress_dataset_bar.update(1)

        progress_dataset_bar.close()

        return videos
        



if __name__=="__main__":
    train_dataset = Sakuga_Dataset(
        instance_data_root='',
        height= 480,
        width=  720,
        video_reshape_mode="center",
        fps=8,
        max_num_frames=49,
        skip_frames_start=0,
        skip_frames_end=0,
        cache_dir="~/.cache",
        id_token="",
        data_information="../../../Datasets/SakugaDataset/parquet/fliter_59_aesthetic_precise.parquet"
    )
    data=train_dataset.__getitem__(0)
    print(data["instance_video"].shape)