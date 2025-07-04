# LongAnimation: Long Animation Generation with Dynamic Global-Local Memory
<a href="https://cn-makers.github.io/long_animation_web/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/pdf/2507.01945"><img src="https://img.shields.io/badge/arXiv-2057.01945-b31b1b.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>


https://github.com/user-attachments/assets/a3866f82-b07a-41ae-9673-2a24f7c78af4



> <a href="https://cn-makers.github.io/long_animation_web/">**LongAnimation: Long Animation Generation with Dynamic Global-Local Memory**</a>
>

[Nan Chen](https://openreview.net/profile?id=~Nan_Chen13)<sup>1</sup>, [Mengqi Huang](https://corleone-huang.github.io/)<sup>1</sup>, [Yihao Meng](https://yihao-meng.github.io/)<sup>2</sup>,  [Zhendong Mao](https://faculty.ustc.edu.cn/maozhendong/en/index.htm)<sup>†,1</sup><br>
<sup>1</sup>USTC <sup>2</sup>HKUST <sup>†</sup>corresponding author

> Existing	studies	are	limited	to	short-term	colorization	by	fusing	overlapping	features	to	achieve	smooth	transitions,	which	fails	to maintain	long-term	color	consistency.	In	this	study,	we	propose	a	dynamic	global-local	paradigm	to	achieve	ideal	long-term	color consistency	by	dynamically	extracting	global	color-consistent	features	relevant	to	the	current	generation.	
</p>

🎉🎉 Our paper, “LongAnimation: Long Animation Generation with Dynamic Global-Local Memory” accepted by ICCV 2025!
**Strongly recommend seeing our [demo page](https://cn-makers.github.io/long_animation_web/).**


## Showcase

https://github.com/user-attachments/assets/8d225a9e-6e27-42bd-9638-5f4e4cb4dbf7

https://github.com/user-attachments/assets/0fee3eed-8a38-4382-bbe6-21c0cf2371e9

https://github.com/user-attachments/assets/7d87e63a-f5e6-46ba-bb1b-d457ceb0b1d8


## Creative usage
### Text-guided Background Generation
https://github.com/user-attachments/assets/68a5d0fb-f767-4fc8-aed6-cd798301484f

https://github.com/user-attachments/assets/7cba4d5b-b793-474d-9da4-34892853b240

https://github.com/user-attachments/assets/6787349b-6a3e-4ed1-8a6a-efc1643a4e92
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
```
@misc{chen2025longanimationlonganimationgeneration,
      title={LongAnimation: Long Animation Generation with Dynamic Global-Local Memory}, 
      author={Nan Chen and Mengqi Huang and Yihao Meng and Zhendong Mao},
      year={2025},
      eprint={2507.01945},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.01945}, 
}
```