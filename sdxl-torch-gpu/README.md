# SDXL

This tool uses the stability AI stable diffusion XL base v1.0 model to render images.

## Requirements

This tool requires either an NVIDIA GPU or Mac with Apple silicon chip to run.
You will also need 9-16GB of RAM available to the GPU to run the tool.

The first invocation of this tool will download several gigabytes from the internet for the model.
It will also install the necessary dependencies.

## Usage

To use this tool, request AI to render an image and provide a URL.

As mentioned before, a large amount of data is downloaded in the first run. Rendering time is also varied based on the machine's GPU and RAM.

A simple example:

```gptscript
tools: github.com/cloudnautique/local-image-gen/sdxl-torch-gpu

Draw a rooster on a farm at sunset.
```

Will create something like:

![rooster](https://github.com/cloudnautique/local-image-gen/blob/main/sdxl-torch-gpu/assets/rooster.png?raw=true)
