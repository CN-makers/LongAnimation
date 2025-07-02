
def encode_video(video):
    video = video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0)
    
    #print(video.shape)

    video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    image = video[:, :, :1].clone()
    with torch.no_grad():
        latent_dist = vae.encode(video).latent_dist
    
    #为什么训练的时候不引入参考图像呢？而是使用纯噪声，并且这个噪声的均值是-3，方差是0.5,例如tensor([0.0549])
    image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=image.device)
    image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=image.dtype)
    noisy_image = torch.randn_like(image) * image_noise_sigma[:, None, None, None, None]
    image_latent_dist = vae.encode(noisy_image).latent_dist

    return latent_dist, image_latent_dist

