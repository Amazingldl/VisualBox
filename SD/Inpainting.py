import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler


def remove(in_path, out_path):
    image = Image.open(in_path)
    prefix, _ = in_path.rsplit(".", 1)
    mask_image = Image.open(prefix + "_expand.png")
    prompt = "background"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        # "./model/SD/stable-diffusion-inpainting",
        "./model/SD/stable-diffusion-2-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

    w, h = image.size
    h = h // 8 * 8
    w = w // 8 * 8
    removed_img = pipe(prompt=prompt, image=image, mask_image=mask_image, height=h, width=w).images[0]
    removed_img.save(out_path)

