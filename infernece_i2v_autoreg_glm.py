
import argparse
import logging
import math
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
    load_image,
)
from torchvision.transforms import ToPILImage
import torch
from pathlib import PosixPath


from utils.utils import load_model_from_config,load_segmented_safe_weights,control_weight_files

from models.cogvideox_transformer_3d_control import Control3DModel,Controled_CogVideoXTransformer3DModel
from models.pipeline_cogvideox_image2video import Controled_CogVideoXImageToVideoPipeline,Controled_Memory_CogVideoXImageToVideoPipeline
from models.global_local_memory_module import global_local_memory
import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    #CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from lineart_extractor.annotator.lineart import LineartDetector
from diffusers.image_processor import VaeImageProcessor

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import (
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
import numpy as np
from videoxl.model.builder import load_pretrained_model
from videoxl.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videoxl.constants import IMAGE_TOKEN_INDEX,TOKEN_PERFRAME 
try:
    import decord
except ImportError:
    raise ImportError(
        "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
    )
decord.bridge.set_bridge("torch")
from utils.autoreg_video_save_function import autoreg_video_save
from decord import VideoReader, cpu
from einops import rearrange
import gc


def _resize_for_rectangle_crop(arr,height,width,video_reshape_mode):
    image_size = height,width
    reshape_mode = video_reshape_mode
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
    image_size = height, width
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

def get_frame_length(frame_path):
    video_reader = decord.VideoReader(uri = frame_path.as_posix())
    video_num_frames = len(video_reader)
    return video_num_frames

def proccess_frame(frame_path,frames_start,frames_end):
    
    video_reader = decord.VideoReader(uri = frame_path.as_posix())
    video_num_frames = len(video_reader)

    start_frame = frames_start
    end_frame = frames_end
    
    indices = list(range(start_frame, end_frame))
    frames = video_reader.get_batch(indices)
    
    #frames = frames[start_frame: end_frame]
    selected_num_frames = frames.shape[0]
    print("selected_num_frames",selected_num_frames)
    # Choose first (4k + 1) frames as this is how many is required by the VAE
    remainder = (3 + (selected_num_frames % 4)) % 4
    if remainder != 0:
        frames = frames[:-remainder]
    selected_num_frames = frames.shape[0]

    assert (selected_num_frames - 1) % 4 == 0

    # Training transforms
    
    frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
    frames = _resize_for_rectangle_crop(frames,height=args.height,width=args.width,video_reshape_mode="center")
    final_frames = frames.contiguous()
    return final_frames


def proccess_image(frames):

    # Training transforms
    
    frames = frames.unsqueeze(0).permute(0, 3, 1, 2) # [F, C, H, W]
    frames = _resize_for_rectangle_crop(frames,height=args.height,width=args.width,video_reshape_mode="center")
    final_frames = frames.contiguous()
    return final_frames



def encode_sketch(video,pipe):

    
    video = video.to(pipe.vae.device, dtype=pipe.vae.dtype).unsqueeze(0)
    video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    
    latent_dist = pipe.vae.encode(video).latent_dist
    return latent_dist

def process_sketch(sketch,linear_detector,pipe):
    sketch = sketch.to("cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        sketch =  linear_detector(sketch,coarse=False)

    sketch=(sketch>0.78).float()
    sketch=1-sketch
    sketch=sketch.repeat(1,3,1,1)
    sketch = (sketch - 0.5) / 0.5
    sketch=sketch.contiguous()

    sketch = sketch.to(pipe.vae.device, dtype=pipe.vae.dtype).unsqueeze(0)
    sketch = sketch.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    image = sketch[:, :, :1].clone()

    with torch.no_grad():
        sketch = pipe.vae.encode(sketch).latent_dist
        sketches_first_frame=pipe.vae.encode(image).latent_dist

    sketch = sketch.sample() * pipe.vae.config.scaling_factor
    sketches_first_frame= sketches_first_frame.sample() * pipe.vae.config.scaling_factor


    sketch = sketch.permute(0, 2, 1, 3, 4)
    sketch =  sketch.to(memory_format=torch.contiguous_format)
    
    sketches_first_frame = sketches_first_frame.permute(0, 2, 1, 3, 4)
    sketches_first_frame = sketches_first_frame.to(memory_format=torch.contiguous_format)



    return sketch,sketches_first_frame



def process_sketch_image(sketch,linear_detector,pipe):
    
    
    sketch=torch.tensor(np.array(sketch))
    sketch=proccess_image(sketch)

    sketch = sketch.to("cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        sketch =  linear_detector(sketch,coarse=False)


    sketch=(sketch>0.78).float()
    sketch=1-sketch
    sketch=sketch.repeat(1,3,1,1)
    sketch = (sketch - 0.5) / 0.5
    sketch=sketch.contiguous()

    sketch = sketch.to(pipe.vae.device, dtype=pipe.vae.dtype).unsqueeze(0)
    sketch = sketch.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    

    with torch.no_grad():
        sketch = pipe.vae.encode(sketch).latent_dist
       

    sketch = sketch.sample() * pipe.vae.config.scaling_factor

    sketch = sketch.permute(0, 2, 1, 3, 4)
    sketch =  sketch.to(memory_format=torch.contiguous_format)
    return sketch


def log_validation(
    pipe,
    args,
    pipeline_args,
    device,
    use_glm=False,
    global_memory=None,
    local_memory=None,
    glm=None,
    past_latents=None,
    
    

):

   
    scheduler_args = {}
    idx = pipeline_args.pop("segment", None)
    video_key=pipeline_args.pop("video_key", None)
    clip_memory=False if idx==0 else True
    print("clip_memory",clip_memory)

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None

    videos = []
    os.makedirs(os.path.join(args.output_dir,video_key),exist_ok=True)
    video_tensor_path=os.path.join(args.output_dir,video_key)
    print(video_tensor_path,"video_tensor_path")
    
    with torch.no_grad():
        for _ in range(args.num_validation_videos):
            
            frames_output, past_latents = pipe(**pipeline_args, generator=generator, output_type="pt",
                                num_inference_steps=50,use_glm=use_glm,
                                global_memory=global_memory,
                                local_memory=local_memory,
                                glm=glm,
                                video_tensor_path=video_tensor_path,
                                past_latents=past_latents[:,-4:-2] if (past_latents is not None) else None ,
                                clip_memory=clip_memory 
                                )
            pt_images=frames_output.frames[0]
            #TODO here we can choose if we need the first frame or not
            pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])

            image_np = VaeImageProcessor.pt_to_numpy(pt_images)
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)

            videos.append(image_pil)

    phase_name =  f"inference_{idx}"
    video_filenames = []
    for i, video in enumerate(videos):
        
        final_output_dir=os.path.join(args.output_dir,video_key)
        os.makedirs(final_output_dir,exist_ok=True)
        
        filename = os.path.join(final_output_dir, f"{phase_name}_video.mp4")

        export_to_video(video, filename, fps=args.fps)
        video_filenames.append(filename)
        
        autoreg_video_save(base_path=final_output_dir,suffix="inference_{}_video.mp4",num_videos=idx+1)
        

    return videos[0][65]

def save_segments(total_frames,segment_length,overlap):
    start_frame = 0
    segments = []
    while start_frame + segment_length <= total_frames:
        end_frame = start_frame + segment_length
        segments.append((start_frame, end_frame))
        start_frame = end_frame - overlap
        
    return segments
    
    
    
    

def main(args):

    os.makedirs(args.output_dir,exist_ok=True)
    load_dtype=torch.bfloat16
    transformer =Controled_CogVideoXTransformer3DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=load_dtype,
            )

    control_config_path = "model_json/control_model_15_small.json"
    transformer_control_config = load_model_from_config(control_config_path)
    transformer_control = Control3DModel(**transformer_control_config)
    control_weight_files=[args.control_weght]
    transformer_control = load_segmented_safe_weights(transformer_control, control_weight_files)
    
    transformer_control = transformer_control.to(load_dtype)

    linear_detector=LineartDetector("cuda", dtype=torch.bfloat16)

    gen_kwargs = {"do_sample": True, "temperature": 1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 2}

    # try:
    #     video_tokenizer, video_model, clip_image_processor, _ = load_pretrained_model(args.llm_model_path, None, "llava_qwen", device_map="cuda",attn_implementation="flash_attention_2")
    # except:
    video_tokenizer, video_model, clip_image_processor, _ = load_pretrained_model(args.llm_model_path, None, "llava_qwen", device_map="cuda",attn_implementation="sdpa")
    video_model.config.beacon_ratio=[8]   # you can delete this line to realize random compression of {2,4,8} ratio
    vllm_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nCan you describe the scene and color in anime?<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer_image_token(vllm_prompt, video_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(video_model.device)
    video_model.to( dtype=torch.bfloat16)

    
    glm=global_local_memory()
    glm_weight_files=[args.glm_weight]
    glm = load_segmented_safe_weights(glm,glm_weight_files)
    glm=glm.to(load_dtype)
    glm=glm.to("cuda")
    print("successful load glm")
    
    


    pipe = Controled_Memory_CogVideoXImageToVideoPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        transformer=transformer,
        transformer_control=transformer_control
        ).to("cuda")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    del transformer,transformer_control
    gc.collect()
    torch.cuda.empty_cache()
    

    #pipe.enable_sequential_cpu_offload()
    
    if args.enable_slicing:
        pipe.vae.enable_slicing()
    if args.enable_tiling:
        pipe.vae.enable_tiling()
    


    #pipe = pipe.to("cuda")

    

    import json

    
    with open('test_json/long_testset.json',"r") as json_file:
        video_info=json.load(json_file)
    
    
    for video_key,value in video_info.items():
        print('------------')
        print(video_key)
        validation_prompt=value['prompt']
        video_path=PosixPath(value['video_path'])
        
        reference_image_path=str(value["reference_image"])

        
        use_glm=False
        
        i=0
        global_image=None
           
       
        frame_path = video_path
        video_num_frames=get_frame_length(frame_path)
        segments=save_segments(total_frames=video_num_frames,segment_length=args.max_num_frames,overlap=16)
        print(segments)

        ''''''
        past_latents=None
        for seg_idx,segment in enumerate(segments):
            
            print(seg_idx)
            print(segment)
            videos = proccess_frame(frame_path, frames_start=segment[0], frames_end=segment[1])
            #print(segment)
            
            sketches,sketches_first_frame = process_sketch(videos,linear_detector,pipe)
            torch.cuda.empty_cache()
            print("sketches!!!",sketches.shape)
            
            validation_prompt = validation_prompt+" High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
            
            to_pil=ToPILImage()
            

            if global_image==None:
                print("------------------")
                print(reference_image_path)
                print('------------------')
                if reference_image_path != "0":
                    image=Image.open(reference_image_path).convert("RGB")
                    global_image=image
                   
                    sketches_first_frame = process_sketch_image(global_image,linear_detector,pipe)
                
                else:
                    image=to_pil(videos[0]).convert("RGB")
                    global_image=image
                    
                    sketches_first_frame = process_sketch_image(global_image,linear_detector,pipe)
            else:
                image=global_image


            pipeline_args = {
                "image": image,
                "prompt": validation_prompt,
                "guidance_scale": args.guidance_scale,
                "use_dynamic_cfg": args.use_dynamic_cfg,
                "height": args.height,
                "width": args.width,
                "sketches": sketches,
                "sketches_first_frame":sketches_first_frame,
                "num_frames":args.max_num_frames,
                "segment": seg_idx,
                "video_key":video_key
            }
            #load the video and process the video

            if use_glm:
                auto_path=os.path.join(args.output_dir,video_key,"autoreg_video_1.mp4")

                vr = VideoReader(auto_path, ctx=cpu(0))
                total_frame_num = len(vr)
                if total_frame_num>650:
                    max_frame=650
                else:
                    max_frame=total_frame_num
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frame, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frames = vr.get_batch(frame_idx).numpy()
                print(frames.shape)
               
                global_videos = clip_image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(video_model.device, dtype=torch.bfloat16)
                local_videos=global_videos[-20:,]
                
                
                beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()

                with torch.inference_mode():
                    
                    num_tokens=TOKEN_PERFRAME *global_videos.shape[0]
                    beacon_skip_last = beacon_skip_first  + num_tokens
                    
                    video_model.generate(input_ids, images=[global_videos],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)
                    indices=[-9,-5,-1]
                    global_memory=torch.cat([
                        torch.cat([rearrange(video_model.past_key_values[i][0], 'b c h w -> b h (c w)') for i in indices],dim=0).unsqueeze(0), 
                        torch.cat([rearrange(video_model.past_key_values[i][1], 'b c h w -> b h (c w)') for i in indices],dim=0).unsqueeze(0)] 
                    ,dim=0).unsqueeze(0)
                    video_model.clear_past_key_values()
                    video_model.memory.reset()
                    print(global_memory.shape)
                    torch.cuda.empty_cache()
                    
                    num_tokens=TOKEN_PERFRAME *local_videos.shape[0]
                    beacon_skip_last = beacon_skip_first  + num_tokens
                    video_model.generate(input_ids, images=[local_videos],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)
                   
                    indices=[-9,-5,-1]
                    local_memory=torch.cat([
                        torch.cat([rearrange(video_model.past_key_values[i][0], 'b c h w -> b h (c w)') for i in indices],dim=0).unsqueeze(0), 
                        torch.cat([rearrange(video_model.past_key_values[i][1], 'b c h w -> b h (c w)') for i in indices],dim=0).unsqueeze(0)] 
                    ,dim=0).unsqueeze(0)
                    video_model.clear_past_key_values()
                    video_model.memory.reset()
                    del global_videos,local_videos
                    torch.cuda.empty_cache()
                    print(local_memory.shape)
                
            else:
                global_memory=None
                local_memory=None
    
            last_image=log_validation(
                pipe=pipe,
                args=args,
                pipeline_args=pipeline_args,
                device="cuda",
                use_glm=use_glm,
                global_memory=global_memory,
                local_memory=local_memory,
                glm=glm,
                past_latents=past_latents
            )
            torch.cuda.empty_cache()
            use_glm=True
                

    
def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--llm_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--control_weght",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--glm_weight",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )



    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-i2v-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=81, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    return parser.parse_args()



if __name__=="__main__":
    args = get_args()
    main(args)


