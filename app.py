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


# freddyaboulton/dracula_revamped
# bethecloud/storj_theme
with gr.Blocks(title="Visual Box", theme="freddyaboulton/dracula_revamped") as demo:
    state_points = gr.State([])
    state_labels = gr.State([])
    state_masks = gr.State([])
    gr.Markdown(""
                "## Visual Box"
                "")

    with gr.Tab("分割 & 编辑 & 移除"):
        with gr.Row():
            label_type = gr.Radio(label="鼠标选择类型", choices=["1", "0"], value="1", interactive=True)
        gr.Markdown("分割")
        with gr.Row():
            input_img = gr.Image(label="原始图像", type="numpy")
            seg_img = gr.Image(label="分割图像", interactive=False)
        mask_img = gr.Image(label="mask image", image_mode="L", interactive=False, type="numpy", visible=False)
        expand_img = gr.Image(label="expand mask", image_mode="L", interactive=False, visible=False)
        gr.Markdown("编辑")
        with gr.Row():
            with gr.Column():
                sd_inpaint_text = gr.Textbox(label="提示词", lines=3, interactive=True)
                with gr.Row():
                    sd_clear_btn = gr.Button("清除文本")
                    sd_inpaint_btn = gr.Button("通过SD模型生成图像", variant="primary")
            with gr.Column():
                sd_inpaint_img = gr.Image(label="SD生成图像", interactive=False)
        gr.Markdown("移除")
        with gr.Row():
            with gr.Column():
                lama_removed_btn = gr.Button("通过lama模型移除对象", variant="primary")
            lama_removed_img = gr.Image(label="lama生成图像", interactive=False)
        # with gr.Row():
        #     clear_btn1 = gr.Button("清除所有图像")
            # mask_btn = gr.Button("Generate mask image")
            # expand_btn = gr.Button("Expand mask")
            # sd_removed_btn = gr.Button("通过SD模型生成图像", variant="primary")
            # lama_removed_btn = gr.Button("通过lama模型移除对象", variant="primary")

    with gr.Tab("文本生成图像"):
        with gr.Row():
            with gr.Column():
                t2i_style = gr.Radio(choices=["传统中式", "新中式", "日式侘寂风", "北欧清新风", "现代极简风"], label="装修风格")
                t2i_prompt = gr.Textbox(label="正向提示词", lines=3, interactive=True)
                t2i_nprompt = gr.Textbox(label="反向提示词", lines=3, interactive=True)
                t2i_width = gr.Slider(label="图像宽度", minimum=512, maximum=1536, value=1024, step=4, interactive=True)
                t2i_height = gr.Slider(label="图像高度", minimum=512, maximum=1536, value=576, step=4, interactive=True)
                t2i_seed = gr.Number(label="随机种子(-1表示随机)", value=-1, interactive=True)
                t2i_img_nums = gr.Number(label="生成数量", value=1, interactive=True)
                with gr.Row():
                    t2i_clear_btn = gr.Button("清除文本")
                    t2i_gene_btn = gr.Button("生成", variant="primary")
            # t2i_img = gr.Image(interactive=False)
            t2i_gallery = gr.Gallery(
                label="Generated image", show_label=False, elem_id="gallery"
                ).style(columns=[4], rows=[2], object_fit="contain", height="auto")
            
    with gr.Tab("图像生成图像"):
        with gr.Row():
            with gr.Column():
                i2i_style = gr.Radio(choices=["传统中式", "新中式", "日式侘寂风", "北欧清新风", "现代极简风"], label="装修风格")
                i2i_img = gr.Image(label="原始图像")
                i2i_prompt = gr.Textbox(label="正向提示词", lines=3, interactive=True)
                i2i_nprompt = gr.Textbox(label="反向提示词", lines=3, interactive=True)
                i2i_seed = gr.Number(label="随机种子(-1表示随机)", value=-1, interactive=True)
                i2i_img_nums = gr.Number(label="生成数量", value=1, interactive=True)
                with gr.Row():
                    i2i_clear_btn = gr.Button("清除")
                    i2i_gene_btn = gr.Button("生成", variant="primary")
            # t2i_img = gr.Image(interactive=False)
            i2i_gallery = gr.Gallery(
                label="Generated image", show_label=False, elem_id="gallery"
                ).style(columns=[4], rows=[2], object_fit="contain", height="auto")

    with gr.Tab("超分辨率"):
        with gr.Row():
            with gr.Column():
                reso_input_img = gr.Image(label="原始图像")
                reso_ratio = gr.Radio(choices=[2, 4], label="放大比例", value=2)
                with gr.Row():
                    reso_clear_btn = gr.Button("清除")
                    reso_gene_btn = gr.Button("生成", variant="primary")
            reso_gene_img = gr.Image(label="超分辨率图像", interactive=False)
    

    with gr.Tab("移除背景"):
        gr.Markdown("自动移除背景")
        with gr.Row():
            with gr.Column():
                rmbg_dis_input_img = gr.Image(label="原始图像", type="filepath")
                with gr.Row():
                    rmbg_dis_clear_btn = gr.Button("清除")
                    rmbg_dis_gene_btn = gr.Button("移除背景", variant="primary")
            rmbg_dis_gene_img = gr.Image(label="移除背景图像", interactive=False)
        gr.Markdown("手动移除背景")
        with gr.Row():
            with gr.Column():
                label_type = gr.Radio(label="鼠标选择类型", choices=["1", "0"], value="1", interactive=True)
                input_img = gr.Image(label="原始图像", type="numpy")
                with gr.Row():
                    rmbg_sam_clear_btn = gr.Button("清除")
                    rmbg_sam_gene_btn = gr.Button("移除背景", variant="primary")
            with gr.Column():
                seg_img = gr.Image(label="分割图像", interactive=False)
                mask_img = gr.Image(label="mask image", image_mode="L", interactive=False, type="numpy", visible=False)
                rmbg_sam_gene_img = gr.Image(label="移除背景图像", interactive=False)

        

    # table 1
    input_img.change(set_image, inputs=[input_img, state_points, state_labels, state_masks],
                     outputs=[state_points, state_labels, state_masks])
    input_img.select(gene_seg, inputs=[input_img, label_type, state_points, state_labels],
                     outputs=[seg_img, state_points, state_labels, state_masks]).success(gene_mask, inputs=[state_masks], outputs=[mask_img]).success(gene_expand, inputs=[mask_img], outputs=[expand_img])
    sd_clear_btn.click(lambda: "", None, [sd_inpaint_text])
    # clear_btn1.click(lambda: [None] * 6, None, [input_img, seg_img, mask_img, expand_img, sd_removed_img, lama_removed_img])
    sd_inpaint_btn.click(gene_sd_inpaint, inputs=[sd_inpaint_text, input_img, expand_img], outputs=[sd_inpaint_img])
    lama_removed_btn.click(gene_lama_removed, inputs=[input_img, expand_img], outputs=[lama_removed_img])
    
    # table 2
    t2i_clear_btn.click(lambda: [None] * 2, None, [t2i_prompt, t2i_nprompt])
    t2i_gene_btn.click(text2img, inputs=[t2i_style, t2i_prompt, t2i_nprompt, t2i_width, t2i_height, t2i_seed, t2i_img_nums], 
                       outputs=[t2i_gallery])
    # table 3
    i2i_clear_btn.click(lambda: [None] * 3, None, [i2i_img, i2i_prompt, i2i_nprompt])
    i2i_gene_btn.click(img2img, inputs=[i2i_style, i2i_img, i2i_prompt, i2i_nprompt, i2i_seed, i2i_img_nums], 
                       outputs=[i2i_gallery])

    # table 4
    reso_clear_btn.click(lambda: [None] * 2, None, [reso_input_img, reso_gene_img])
    reso_gene_btn.click(real_esrgan_app, inputs=[reso_input_img, reso_ratio], outputs=[reso_gene_img])

    # table 5
    rmbg_dis_clear_btn.click(lambda: [None] * 2, None, [rmbg_dis_input_img, rmbg_dis_gene_img])
    rmbg_dis_gene_btn.click(dis_background_remove, inputs=[rmbg_dis_input_img], outputs=[rmbg_dis_gene_img])
    rmbg_sam_clear_btn.click(lambda: [None] * 3, None, [input_img, seg_img, rmbg_sam_gene_img])
    rmbg_sam_gene_btn.click(sam_background_remove, inputs=[input_img, mask_img], outputs=[rmbg_sam_gene_img])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1")
