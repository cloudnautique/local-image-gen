import torch
import os
import sys

from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
)

# Read values from environment variables
prompt = os.getenv(
    "prompt", "A cute puppy, photograph, close up, hyper detailed, intricate details"
)
neg_prompt = os.getenv(
    "neg_prompt",
    "extra fingers, mutated hands, deformed, blurry, bad anatomy, extra limbs, mangled fingers, distorted face, detached body parts, blurry",
)
batch_size = int(os.getenv("batch_size", "1"))

config = {
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "refiner_model": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "use_ensemble_of_experts": True,
    "num_inference_steps": 20,
    "num_refinement_steps": 25,
    "num_images_per_prompt": batch_size,
    "high_noise_fraction": 0.8,
    "guidance_scale": 0.5,
    "scheduler_args": {
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "interpolation_type": "linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "steps_offset": 1,
        "timestep_spacing": "leading",
        "trained_betas": None,
        "use_karras_sigmas": False,
    },
}

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using CUDA...")
# Check for MPS availability if CUDA is not available
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available. Using MPS...")
# Exit if neither CUDA nor MPS is available
else:
    print(
        "CUDA or MPS is not available. Need one of these to use this tool.\nExiting..."
    )
    sys.exit(1)

pipe = DiffusionPipeline.from_pretrained(
    config["model"],
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    scheduler=EulerDiscreteScheduler(**config["scheduler_args"]),
)
pipe.to(device)

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    config["refiner_model"],
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    scheduler=EulerDiscreteScheduler(**config["scheduler_args"]),
)
refiner.to(device)

generator = torch.Generator(device=device)
latent = pipe(
    prompt=prompt,
    neg_prompt=neg_prompt,
    output_type="latent",
    num_images_per_prompt=config["num_images_per_prompt"],
    num_inference_steps=config["num_inference_steps"],
    denoising_end=config["high_noise_fraction"],
    generator=generator,
).images

image = refiner(
    prompt=prompt,
    neg_prompt=neg_prompt,
    guidance_scale=config["guidance_scale"],
    image=latent,
    num_images_per_prompt=config["num_images_per_prompt"],
    num_inference_steps=config["num_refinement_steps"],
    denoising_start=config["high_noise_fraction"],
    generator=generator,
).images

for i in range(0, len(image)):
    image[i].save(f"output-{i}.png")
    print(f"file://{os.getcwd()}/output-{i}.png")
