# LongAnimation: Long Animation Generation with Dynamic Global-Local Memory
<a href="https://cn-makers.github.io/CustomContrast/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/pdf/"><img src="https://img.shields.io/badge/arXiv-1111.11111-b31b1b.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>


video\video.mp4




> <a href="https://cn-makers.github.io/long_animation_web/">**LongAnimation: Long Animation Generation with Dynamic Global-Local Memory**</a>
>

[Nan Chen](https://cn-makers.github.io/)<sup>1</sup>, [Mengqi Huang](https://ken-ouyang.github.io/)<sup>1</sup>, [Yihao Meng](https://openreview.net/profile?id=~Hanlin_Wang2)<sup>2</sup>,  [Zhendong Mao](http://www.huamin.org/index.htm/)<sup>†,1</sup><br>
<sup>1</sup>USTC <sup>2</sup>HKUST <sup>†</sup>corresponding author

> Existing	studies	are	limited	to	short-term	colorization	by	fusing	overlapping	features	to	achieve	smooth	transitions,	which	fails	to maintain	long-term	color	consistency.	In	this	study,	we	propose	a	dynamic	global-local	paradigm	to	achieve	ideal	long-term	color consistency	by	dynamically	extracting	global	color-consistent	features	relevant	to	the	current	generation.	
</p>

**Strongly recommend seeing our [demo page](https://cn-makers.github.io/long_animation_web/).**


## Showcase
<p style="text-align: center;">
  <img src="video\text_1.mp4" alt="GIF" />
</p>
<p style="text-align: center;">
  <img src="video\text_2.mp4" alt="GIF" />
</p>
<p style="text-align: center;">
  <img src="video\text_3.mp4" alt="GIF" />
</p>

## Creative usage
### Text-guided Background Generation
<div style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
<img src="video\text_1.mp4" alt="GIF Animation">
<img src="video\text_2.mp4" alt="GIF Animation">
<img src="video\text_3.mp4" alt="GIF Animation"  style="margin-bottom: 40px;"> 
<div style="text-align:center; margin-top: -50px; margin-bottom: 70px;font-size: 18px; letter-spacing: 0.2px;">
        <em>A boy and a girl in different environment.</em>
</div>
</div>

## TODO List

- [x] Release the paper and demo page. Visit [https://cn-makers.github.io/long_animation_web/](https://cn-makers.github.io/long_animation_web/) 
- [x] Release the code.


## Requirements
The training is conducted on 6 A100 GPUs (80GB VRAM), the inference is tested on 1 A100 GPU. 
## Setup
```
git clone https://github.com/CN-makers/LongAnimation
cd LongAnimation
```

## Environment
All the tests are conducted in Linux. We suggest running our code in Linux. To set up our environment in Linux, please run:
```
conda create -n LongAnimation python=3.10 -y
conda activate LongAnimation

bash install.sh
```


## Checkpoints
1. please download the pre-trained CogVideoX-1.5 I2V  checkpoints from [here](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V), and put the whole folder under `pretrained_weight`, it should look like `./pretrained_weights/CogVideoX1.5-5B-I2V`

2. please download the pre-trained long video understanding model Video-XL  checkpoints from [here](https://huggingface.co/sy1998/Video_XL/tree/main), and put the whole folder under `pretrained_weight`, it should look like `./pretrained_weights/videoxl`

3. please download the checkpoint for our SketchDiT and DGLM model from [here](https://huggingface.co/CNcreator0331/LongAnimation/tree/main), and put the whole folder as `./pretrained_weights/longanimation`.

   



## Generate Your Animation!
To colorize the target lineart sequence with a specific character design, you can run the following command:
```
bash  long_animation_inference.sh
```


We provide some test cases in  `test_json` folder. You can also try our model with your own data. You can change the lineart sequence and corresponding character design in the script `Long_animation_inference.sh`.

During the official training and testing, the --height and --weight we used were 576 and 1024 respectively. Additionally, the model can also be compatible with resolutions of 768 in length and 1360 in width respectively.



## Citation:
Don't forget to cite this source if it proves useful in your research!
