import gradio as gr
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from util import *


sam_checkpoint = "./model/SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def set_image(img, points, labels, masks):  # 只有加载和卸载图片时会调用
    if img is not None:

        predictor.set_image(img)
        # print("set")
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



with gr.Blocks(title="Visual Box", theme="bethecloud/storj_theme") as demo:
    state_points = gr.State([])
    state_labels = gr.State([])
    state_masks = gr.State([])
    gr.Markdown(""
                "## Visual Box"
                "")

    with gr.Tab("Seg & Remove"):
        with gr.Row():
            label_type = gr.Radio(label="label type", choices=["1", "0"], value="1", interactive=True)
        with gr.Row():
            input_img = gr.Image(label="input image", type="numpy")
            seg_img = gr.Image(label="seg image", interactive=False)
            mask_img = gr.Image(label="mask image", image_mode="L", interactive=False, type="numpy", visible=True)
            expand_img = gr.Image(label="expand mask", image_mode="L", interactive=False, visible=True)
        with gr.Row():
            sd_removed_img = gr.Image(label="sd removed image", interactive=False)
            lama_removed_img = gr.Image(label="lama removed image", interactive=False)
        with gr.Row():
            clear_btn1 = gr.Button("Clear all")
            # mask_btn = gr.Button("Generate mask image")
            # expand_btn = gr.Button("Expand mask")
            sd_removed_btn = gr.Button("Generate sd removed image", variant="primary")
            lama_removed_btn = gr.Button("Generate lama removed image", variant="primary")
    with gr.Tab("Text to Image"):
        with gr.Row():
            with gr.Column():
                t2i_prompt = gr.Textbox(label="prompt", lines=3, interactive=True)
                t2i_nprompt = gr.Textbox(label="negative prompt", lines=3, interactive=True)
                t2i_width = gr.Slider(label="width", minimum=512, maximum=1536, value=1024, step=4, interactive=True)
                t2i_height = gr.Slider(label="height", minimum=512, maximum=1536, value=576, step=4, interactive=True)
                with gr.Row():
                    clear_btn2 = gr.Button("Clear Text")
                    t2i_gene_btn = gr.Button("Generate", variant="primary")
            t2i_img = gr.Image(interactive=False)
    with gr.Tab("Image to Image"):
        pass


    # table 1
    input_img.change(set_image, inputs=[input_img, state_points, state_labels, state_masks],
                     outputs=[state_points, state_labels, state_masks])
    input_img.select(gene_seg, inputs=[input_img, label_type, state_points, state_labels],
                     outputs=[seg_img, state_points, state_labels, state_masks]).success(gene_mask, inputs=[state_masks], outputs=[mask_img]).success(gene_expand, inputs=[mask_img], outputs=[expand_img])
    clear_btn1.click(lambda: [None] * 6, None, [input_img, seg_img, mask_img, expand_img, sd_removed_img, lama_removed_img])
    sd_removed_btn.click(gene_sd_removed, inputs=[input_img, expand_img], outputs=[sd_removed_img])
    lama_removed_btn.click(gene_lama_removed, inputs=[input_img, expand_img], outputs=[lama_removed_img])
    
    # table 2
    clear_btn2.click(lambda: [None] * 2, None, [t2i_prompt, t2i_nprompt])
    t2i_gene_btn.click(text2img, inputs=[t2i_prompt, t2i_nprompt, t2i_width, t2i_height], 
                       outputs=[t2i_img])



if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=8080)


