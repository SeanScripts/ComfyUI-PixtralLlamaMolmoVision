import comfy.utils
import comfy.model_management as mm
import folder_paths

from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, set_seed
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import time
import os
from pathlib import Path

class PixtralModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([item.name for item in Path(folder_paths.models_dir, "pixtral").iterdir() if item.is_dir()],),
            }
        }
    
    RETURN_TYPES = ("PIXTRAL_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Pixtral"
    TITLE = "PixtralModelLoader"
    
    def load_model(self, model_name):
        model_path = os.path.join(folder_paths.models_dir, "pixtral", model_name)
        device = mm.get_torch_device()
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            use_safetensors=True,
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_path)
        pixtral_model = {
            'model': model,
            'processor': processor,
        }
        return (pixtral_model,)

class PixtralGenerateText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixtral_model": ("PIXTRAL_MODEL",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {"default": "<s>[INST]Caption this image:\n[IMG][/INST]", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "Pixtral"
    TITLE = "PixtralGenerateText"
    
    def generate_text(self, pixtral_model, images, prompt, max_new_tokens, do_sample, temperature, seed):
        device = mm.get_torch_device()
        print(type(images))
        # How does batched input work? I really don't know
        # Also I'm sure there is a way to do this without converting back to numpy and then PIL...
        # Pixtral requires PIL input for some reason, and the to_pil_image function requires channels to be the first dimension for a Tensor but the last dimension for a numpy array... Yeah idk
        print(f"Batch of {images.shape} images")
        image_list = [to_pil_image(image.numpy()) for image in images]
        inputs = pixtral_model['processor'](images=image_list, text=prompt, return_tensors="pt").to(device)
        prompt_tokens = len(inputs['input_ids'][0])
        print(f"Prompt tokens: {prompt_tokens}")
        set_seed(seed)
        t0 = time.time()
        generate_ids = pixtral_model['model'].generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        t1 = time.time()
        total_time = t1 - t0
        generated_tokens = len(generate_ids[0]) - prompt_tokens
        time_per_token = generated_tokens/total_time
        print(f"Generated {generated_tokens} tokens in {total_time:.3f} s ({time_per_token:.3f} tok/s)")
        print(len(generate_ids[0][prompt_tokens:]))
        output = pixtral_model['processor'].decode(generate_ids[0][prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output)
        return (output,)

NODE_CLASS_MAPPINGS = {
    "PixtralModelLoader": PixtralModelLoader,
    "PixtralGenerateText": PixtralGenerateText,
    # Not really much need to work with the image tokenization directly for something like image captioning, but might be interesting later...
    #"PixtralImageEncode": PixtralImageEncode,
    #"PixtralTextEncode": PixtralTextEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixtralModelLoader": "PixtralModelLoader",
    "PixtralGenerateText": "PixtralGenerateText",
}
