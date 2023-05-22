import os
current_path = os.path.dirname(__file__)


from OF import of_inference
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, ControlNetModel, \
    StableDiffusionControlNetPipeline, UniPCMultistepScheduler, StableDiffusionUpscalePipeline


from pathlib import Path
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"



def sd_4x(img_path, save_path):
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    low_img = Image.open(img_path)
    upscaled_image = pipeline(image=low_img).images[0]
    upscaled_image.save(save_path)


class StableDiffusionModel(object):
    def __init__(self, repo_id="Indoor"):
        # self.repo_path = Path("./model/SD") / repo_id
        self.repo_path = "./model/SD/" + repo_id
        self.text2img_pipe = StableDiffusionPipeline.from_pretrained(
            self.repo_path,
            torch_dtype=torch.float16,
            # custom_pipeline="lpw_stable_diffusion",
        ).to(device)
        components = self.text2img_pipe.components
        self.text2img_pipe.enable_xformers_memory_efficient_attention()
        self.text2img_pipe.enable_attention_slicing()
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(**components)  # reuse
        self.img2img_pipe.enable_xformers_memory_efficient_attention()
        self.img2img_pipe.enable_attention_slicing()

    def text2img(self, save_path, prompt, n_prompt=None, width=1024, height=576, seed=0, img_nums=1):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        image = self.text2img_pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            width=width,
            height=height,
            num_inference_steps=40,
            generator=generator,
        ).images[0]
        if save_path is None:
            return image
        else:
            image.save(save_path)

    def img2img(self, in_path, save_path, prompt, n_prompt=None):
        in_img = Image.open(in_path)
        image = self.img2img_pipe(
            image=in_img,
            prompt=prompt,
            negative_prompt=n_prompt,
            num_inference_steps=40,
        ).images[0]
        if save_path is None:
            return image
        else:
            image.save(save_path)


class StableDiffusionControlNetModel(object):
    def __init__(self, repo_id="Indoor"):
        self.repo_path = "./model/SD/" + repo_id

    def text2img_seg(self, control_path, save_path, prompt, n_prompt=None, height=None, width=None, save_seg=False, seg_path=None):
        _, control_image = of_inference(control_path)
        if save_seg:
            control_image.save(seg_path)
        # 推理生成
        control_checkpoint = Path("./model/HF") / "sd-controlnet-seg"
        controlnet = ControlNetModel.from_pretrained(control_checkpoint, torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.repo_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        ).to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()

        image = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=n_prompt,
            num_inference_steps=40,
            image=control_image,
        ).images[0]
        image.save(save_path)

