
export CACHE_PATH="~/.cache"
export MODEL_PATH="pretrained_weight/CogVideoX1.5-5B-I2V"
export CONTROL_WEIGHT="pretrained_weight/control_module.safetensors"
export LLM_MODEL_PATH="pretrained_weight/Video_XL/VideoXL_weight_8"
export GLM_WIGHT="pretrained_weight/glm_module.safetensors"
export NEW_OUTPUT_PATH="output/i2v_autoreg_glm_testset"



CUDA_VISIBLE_DEVICES=4  python infernece_i2v_autoreg_glm.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --llm_model_path  $LLM_MODEL_PATH \
  --num_validation_videos 1 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $NEW_OUTPUT_PATH \
  --height 768 \
  --width 1360 \
  --fps 24 \
  --max_num_frames 81 \
  --allow_tf32 \
  --control_weght $CONTROL_WEIGHT \
  --glm_weight $GLM_WIGHT \
  --enable_slicing \
  --enable_tiling  



