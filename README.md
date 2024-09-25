# ComfyUI-PixtralLlamaVision
 For loading and running Pixtral and Llama 3.2 Vision models

Includes four nodes:
- PixtralModelLoader
- PixtralGenerateText
- LlamaVisionModelLoader
- LlamaVisionGenerateText

These should be self-explanatory.
 
Install the latest version of transformers, which has support for Pixtral/Llama Vision models:
`python_embeded\python.exe -m pip install git+https://github.com/huggingface/transformers`

Requires transformers 4.45.0 for Pixtral and 4.46.0 for Llama Vision.

Also install bitsandbytes if you don't have it already:
`python_embeded\python.exe -m pip install bitsandbytes`

You can get a 4-bit quantized version of Pixtral-12B which is compatible with these custom nodes here: [https://huggingface.co/SeanScripts/pixtral-12b-nf4](https://huggingface.co/SeanScripts/pixtral-12b-nf4)

You can get a 4-bit quantized version of Llama-3.2-11B-Vision-Instruct which is compatible with these custom nodes here:
[https://huggingface.co/SeanScripts/Llama-3.2-11B-Vision-Instruct-nf4](https://huggingface.co/SeanScripts/Llama-3.2-11B-Vision-Instruct-nf4)

![Example workflow](pixtral_caption_example.jpg)
