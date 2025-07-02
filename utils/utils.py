import json
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def load_model_from_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    
    return config

def load_segmented_safe_weights(model, weight_files):
    state_dict = model.state_dict()
    
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt") as f:
            part_state_dict = {key: torch.tensor(f.get_tensor(key)) for key in f.keys()}
            state_dict.update(part_state_dict)
    
    model.load_state_dict(state_dict,strict=False)
    return model

def save_model_weights(model, save_path):
    state_dict = model.state_dict()
    #tensors = {key: value for key, value in state_dict.items()}
    save_file(state_dict, save_path)

# control_weight_files = [
# '../../models/CogVideoX-5b-I2V/transformer_control/diffusion_pytorch_model-00001-of-00003.safetensors', 
# '../../models/CogVideoX-5b-I2V/transformer_control/diffusion_pytorch_model-00002-of-00003.safetensors', 
# '../../models/CogVideoX-5b-I2V/transformer_control/diffusion_pytorch_model-00003-of-00003.safetensors', 
# ]

control_weight_files = [
'../../models/CogVideoX1.5-5B-I2V/transformer_control/diffusion_pytorch_model-00001-of-00003.safetensors', 
'../../models/CogVideoX1.5-5B-I2V/transformer_control/diffusion_pytorch_model-00002-of-00003.safetensors', 
'../../models/CogVideoX1.5-5B-I2V/transformer_control/diffusion_pytorch_model-00003-of-00003.safetensors', 
]