import comfy.utils
import comfy.model_management as mm
import folder_paths

from transformers import AutoProcessor, BitsAndBytesConfig, set_seed

pixtral = True
llama_vision = True
# transformers 4.45.0
try:
    from transformers import LlavaForConditionalGeneration
except ImportError:
    print("[ComfyUI-PixtralLlamaVision] Can't load Pixtral, need to update transformers")
    pixtral = False

# transformers 4.46.0
try:
    from transformers import MllamaForConditionalGeneration
except ImportError:
    print("[ComfyUI-PixtralLlamaVision] Can't load Llama Vision, need to update transformers")
    llama_vision = False

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
        device = pixtral_model['model'].device
        print(type(images))
        # I'm sure there is a way to do this without converting back to numpy and then PIL...
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


class LlamaVisionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([item.name for item in Path(folder_paths.models_dir, "llama-vision").iterdir() if item.is_dir()],),
            }
        }
    
    RETURN_TYPES = ("LLAMA_VISION_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "LlamaVision"
    TITLE = "LlamaVisionModelLoader"
    
    def load_model(self, model_name):
        model_path = os.path.join(folder_paths.models_dir, "llama-vision", model_name)
        device = mm.get_torch_device()
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            use_safetensors=True,
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_path)
        llama_vision_model = {
            'model': model,
            'processor': processor,
        }
        return (llama_vision_model,)


class LlamaVisionGenerateText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_vision_model": ("LLAMA_VISION_MODEL",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {"default": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nCaption this image:\n<|image|><|eot_id|><|start_header_id|>assistant<|end_header_id|>", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "LlamaVision"
    TITLE = "LlamaVisionGenerateText"
    
    def generate_text(self, llama_vision_model, images, prompt, max_new_tokens, do_sample, temperature, seed):
        device = llama_vision_model['model'].device
        print(type(images))
        # I'm sure there is a way to do this without converting back to numpy and then PIL...
        # Llama Vision also requires PIL input for some reason, and the to_pil_image function requires channels to be the first dimension for a Tensor but the last dimension for a numpy array... Yeah idk
        print(f"Batch of {images.shape} images")
        image_list = [to_pil_image(image.numpy()) for image in images]
        inputs = llama_vision_model['processor'](images=image_list, text=prompt, return_tensors="pt").to(device)
        prompt_tokens = len(inputs['input_ids'][0])
        print(f"Prompt tokens: {prompt_tokens}")
        set_seed(seed)
        t0 = time.time()
        generate_ids = llama_vision_model['model'].generate(
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

NODE_CLASS_MAPPINGS = {}

if pixtral:
    NODE_CLASS_MAPPINGS |= {
        "PixtralModelLoader": PixtralModelLoader,
        "PixtralGenerateText": PixtralGenerateText,
        # Not really much need to work with the image tokenization directly for something like image captioning, but might be interesting later...
        #"PixtralImageEncode": PixtralImageEncode,
        #"PixtralTextEncode": PixtralTextEncode,
    }

if llama_vision:
    NODE_CLASS_MAPPINGS |= {
        "LlamaVisionModelLoader": LlamaVisionModelLoader,
        "LlamaVisionGenerateText": LlamaVisionGenerateText,
    }

NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}