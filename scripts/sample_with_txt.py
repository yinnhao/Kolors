import os, torch
# from PIL import Image
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model():
    ckpt_dir = '/mnt/ec-data2/ivs/1080p/zyh/kolor/weights/Kolors'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()
    pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False)
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe

def infer(pipe, prompt, save_path):
    image = pipe(
        prompt=prompt,
        height=1280,
        width=720,
        num_inference_steps=50,
        guidance_scale=5.0,
        num_images_per_prompt=1,
        generator= torch.Generator(pipe.device).manual_seed(66)).images[0]
    image.save(save_path)
    
def process_prompts(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pipe = load_model()
    
    with open(input_file, 'r', encoding='utf-8') as file:
        prompts = file.readlines()
    
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()  # 去除行尾的换行符
        if prompt:
            output_path = os.path.join(output_folder, f"image_{i+1}.png")
            infer(pipe, prompt, output_path)
            print(f"Generated image for prompt '{prompt}' at '{output_path}'")


if __name__ == '__main__':
    import fire
    fire.Fire(process_prompts)
