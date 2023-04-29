from SD import StableDiffusionControlNetModel
import warnings
warnings.filterwarnings("ignore")

sdc = StableDiffusionControlNetModel()
sdc.text2img_seg("test.png", "test_control_seg.png", "a living room", save_seg=True, seg_path="test_seg.png")


