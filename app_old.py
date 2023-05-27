import time

import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler
from LAMA import inpaint_img_with_lama

sam_checkpoint = "./model/SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def plt_seg(img, mask, input_point=None, input_label=None, box=None):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    # plt.figure(figsize=(10, 10))
    plt.imshow(img)
    show_mask(mask, plt.gca())
    if input_point is not None:
        show_points(input_point, input_label, plt.gca())
    if box is not None:
        show_box(box, plt.gca())
    plt.axis('off')
    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0)
    img = cv2.imread("temp.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def mask2image(masks):
    for mask in masks:
        h, w = mask.shape[0], mask.shape[1]
        mask_img = Image.new('L', (w, h), 0)  # 灰度图，所有mask图像都是单通道的灰度图
        for i in range(h):
            for j in range(w):
                if mask[i][j]:
                    mask_img.putpixel((j, i), 255)
                else:
                    mask_img.putpixel((j, i), 0)
        return mask_img


def set_image(img, points, labels, masks):  # 只有加载和卸载图片时会调用
    if img is not None:
        predictor.set_image(img)
        print("set")
    else:
        points = []
        labels = []
        masks = []
    return points, labels, masks


def gene_seg(img, label_type, points, labels, evt: gr.SelectData):
    points.append([evt.index[0], evt.index[1]])
    if label_type == "1":
        labels.append(1)
    elif label_type == "0":
        labels.append(0)
    masks, scores, logits = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=False,
    )
    seg_img = plt_seg(img, masks)
    return seg_img, points, labels, masks


def gene_mask(masks):
    return masks[0].astype(np.uint8) * 255


def gene_expand(mask_img):
    mask = mask_img.astype(np.uint8)
    dilate_factor = 20
    dilate_mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return dilate_mask


def gene_sd_removed(image, mask_image):
    prompt = "background"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "./model/SD/stable-diffusion-inpainting",
        # "./model/HF/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

    h, w = image.shape[:2]  # image.shape = (H, W, 3)   mode=RGB
    image = Image.fromarray(image).resize((512, 512))
    mask_image = Image.fromarray(mask_image).resize((512, 512))

    removed_img = pipe(
        prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=7.5,
    ).images[0]
    return removed_img.resize((w, h))


def gene_lama_removed(image, mask_image):
    img_inpainted = inpaint_img_with_lama(image, mask_image, "./LAMA/lama/configs/prediction/default.yaml", "./model/big-lama")
    return img_inpainted



with gr.Blocks(theme='Ajaxon6255/Emerald_Isle', title="Visual Box") as demo:
    state_points = gr.State([])
    state_labels = gr.State([])
    state_masks = gr.State([])
    gr.Markdown(""
                "## Visual Box"
                "")
    with gr.Row():
        label_type = gr.Radio(label="label type", choices=["1", "0"], value="1")
    with gr.Row():
        input_img = gr.Image(label="input image", type="numpy")
        seg_img = gr.Image(label="seg image", interactive=False)
    with gr.Row():
        mask_img = gr.Image(label="mask image", image_mode="L", interactive=False, type="numpy")
        expand_img = gr.Image(label="expand mask", image_mode="L", interactive=False)
    with gr.Row():
        sd_removed_img = gr.Image(label="sd removed image", interactive=False)
        lama_removed_img = gr.Image(label="lama removed image", interactive=False)
    with gr.Row():
        clear_btn = gr.Button("Clear all")
        mask_btn = gr.Button("Generate mask image")
        expand_btn = gr.Button("Expand mask")
        sd_removed_btn = gr.Button("Generate sd removed image", variant="primary")
        lama_removed_btn = gr.Button("Generate lama removed image", variant="primary")

    input_img.change(set_image, inputs=[input_img, state_points, state_labels, state_masks],
                     outputs=[state_points, state_labels, state_masks])
    input_img.select(gene_seg, inputs=[input_img, label_type, state_points, state_labels],
                     outputs=[seg_img, state_points, state_labels, state_masks])
    clear_btn.click(lambda: [None] * 6, None, [input_img, seg_img, mask_img, expand_img, sd_removed_img, lama_removed_img])

    mask_btn.click(gene_mask, inputs=[state_masks], outputs=[mask_img])
    expand_btn.click(gene_expand, inputs=[mask_img], outputs=[expand_img])
    sd_removed_btn.click(gene_sd_removed, inputs=[input_img, expand_img], outputs=[sd_removed_img])
    lama_removed_btn.click(gene_lama_removed, inputs=[input_img, expand_img], outputs=[lama_removed_img])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=8080)
