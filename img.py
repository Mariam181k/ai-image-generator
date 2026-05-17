
!pip install diffusers transformers accelerate safetensors torch gradio -q

import torch
from diffusers import StableDiffusionPipeline
import gradio as gr


pipe = StableDiffusionPipeline.from_pretrained(
    "OFA-Sys/small-stable-diffusion-v0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)


device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)


def generate_image(prompt):

    image = pipe(
        prompt,
        num_inference_steps=20
    ).images[0]

    return image


demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(
        label="Enter Prompt",
        placeholder="Example: A girl wearing white dress"
    ),
    outputs="image",
    title="Real AI Image Generator",
    description="Generate real AI images without API key"
)


demo.launch(share=True)