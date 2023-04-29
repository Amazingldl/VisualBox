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

    old_size = image.size
    image = image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))
    removed_img = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    removed_img = removed_img.resize(old_size)
    removed_img.save(out_path)

