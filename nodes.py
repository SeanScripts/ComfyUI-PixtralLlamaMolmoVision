import comfy.utils
import comfy.model_management as mm
import folder_paths
from transformers import LlavaForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, set_seed
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import time
import os
from pathlib import Path
import re
import logging
import sys

# Set up logging to print to terminal
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Add a print function for immediate output
def debug_print(message):
    print(f"DEBUG: {message}", flush=True)

pixtral_model_dir = os.path.join(folder_paths.models_dir, "pixtral")
llama_vision_model_dir = os.path.join(folder_paths.models_dir, "llama-vision")
# Add pixtral and llama-vision folders if not present
if not os.path.exists(pixtral_model_dir):
    os.makedirs(pixtral_model_dir)
if not os.path.exists(llama_vision_model_dir):
    os.makedirs(llama_vision_model_dir)

class PixtralModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        debug_print(f"Checking for Pixtral models in directory: {pixtral_model_dir}")
        models = []
        if os.path.exists(pixtral_model_dir):
            if any(file.endswith('.safetensors') for file in os.listdir(pixtral_model_dir)):
                models = ['pixtral']  # Use a default name if files are directly in the folder
        debug_print(f"Found Pixtral models: {models}")
        return {
            "required": {
                "model_name": (models,),
            }
        }

    RETURN_TYPES = ("PIXTRAL_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "PixtralLlamaVision/Pixtral"
    TITLE = "Load Pixtral Model"

    def load_model(self, model_name):
        model_path = pixtral_model_dir  # Use the directory itself as the model path
        debug_print(f"Attempting to load Pixtral model from path: {model_path}")
        try:
            device = mm.get_torch_device()
            debug_print(f"Using device: {device}")
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                use_safetensors=True,
                device_map=device,
            )
            debug_print("Pixtral model loaded successfully")
            processor = AutoProcessor.from_pretrained(model_path)
            debug_print("Pixtral processor loaded successfully")
            pixtral_model = {
                'model': model,
                'processor': processor,
            }
            return (pixtral_model,)
        except Exception as e:
            debug_print(f"Error loading Pixtral model: {str(e)}")
            raise

class LlamaVisionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        debug_print(f"Checking for LlamaVision models in directory: {llama_vision_model_dir}")
        models = []
        if os.path.exists(llama_vision_model_dir):
            if any(file.endswith('.safetensors') for file in os.listdir(llama_vision_model_dir)):
                models = ['llama-vision']  # Use a default name if files are directly in the folder
        debug_print(f"Found LlamaVision models: {models}")
        return {
            "required": {
                "model_name": (models,),
            }
        }

    RETURN_TYPES = ("LLAMA_VISION_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "PixtralLlamaVision/LlamaVision"
    TITLE = "Load Llama Vision Model"

    def load_model(self, model_name):
        model_path = llama_vision_model_dir  # Use the directory itself as the model path
        debug_print(f"Attempting to load LlamaVision model from path: {model_path}")
        try:
            device = mm.get_torch_device()
            debug_print(f"Using device: {device}")
            model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                use_safetensors=True,
                device_map=device,
            )
            debug_print("LlamaVision model loaded successfully")
            processor = AutoProcessor.from_pretrained(model_path)
            debug_print("LlamaVision processor loaded successfully")
            llama_vision_model = {
                'model': model,
                'processor': processor,
            }
            return (llama_vision_model,)
        except Exception as e:
            debug_print(f"Error loading LlamaVision model: {str(e)}")
            raise

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
                "temperature": ("FLOAT", {"default": 0.3, "min": 0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "include_prompt_in_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "PixtralLlamaVision/Pixtral"
    TITLE = "Generate Text with Pixtral"

    def generate_text(self, pixtral_model, images, prompt, max_new_tokens, do_sample, temperature, seed, include_prompt_in_output):
        debug_print("Starting Pixtral text generation")
        device = pixtral_model['model'].device
        debug_print(f"Batch of {images.shape} images")
        image_list = [to_pil_image(image.numpy()) for image in images]
        inputs = pixtral_model['processor'](images=image_list, text=prompt, return_tensors="pt").to(device)
        prompt_tokens = len(inputs['input_ids'][0])
        debug_print(f"Prompt tokens: {prompt_tokens}")
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
        debug_print(f"Generated {generated_tokens} tokens in {total_time:.3f} s ({time_per_token:.3f} tok/s)")
        output_tokens = generate_ids[0] if include_prompt_in_output else generate_ids[0][prompt_tokens:]
        output = pixtral_model['processor'].decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        debug_print(f"Generated output: {output}")
        return (output,)

class LlamaVisionGenerateText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_vision_model": ("LLAMA_VISION_MODEL",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {"default": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>Caption this image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "include_prompt_in_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "PixtralLlamaVision/LlamaVision"
    TITLE = "Generate Text with Llama Vision"

    def generate_text(self, llama_vision_model, images, prompt, max_new_tokens, do_sample, temperature, seed, include_prompt_in_output):
        debug_print("Starting LlamaVision text generation")
        device = llama_vision_model['model'].device
        debug_print(f"Batch of {images.shape} images")
        image_list = [to_pil_image(image.numpy()) for image in images]
        inputs = llama_vision_model['processor'](images=image_list, text=prompt, return_tensors="pt").to(device)
        prompt_tokens = len(inputs['input_ids'][0])
        debug_print(f"Prompt tokens: {prompt_tokens}")
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
        debug_print(f"Generated {generated_tokens} tokens in {total_time:.3f} s ({time_per_token:.3f} tok/s)")
        output_tokens = generate_ids[0] if include_prompt_in_output else generate_ids[0][prompt_tokens:]
        output = llama_vision_model['processor'].decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        debug_print(f"Generated output: {output}")
        return (output,)

NODE_CLASS_MAPPINGS = {
    "PixtralModelLoader": PixtralModelLoader,
    "PixtralGenerateText": PixtralGenerateText,
    "LlamaVisionModelLoader": LlamaVisionModelLoader,
    "LlamaVisionGenerateText": LlamaVisionGenerateText,
}

NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}

# Add this at the end of your script to test the loaders
if __name__ == "__main__":
    debug_print("Starting model loader test")
    
    debug_print("Testing Pixtral Model Loader")
    pixtral_loader = PixtralModelLoader()
    pixtral_models = pixtral_loader.INPUT_TYPES()['required']['model_name'][0]
    if pixtral_models:
        debug_print(f"Testing with first available Pixtral model: {pixtral_models[0]}")
        pixtral_loader.load_model(pixtral_models[0])
    else:
        debug_print("No Pixtral models found to test with")
    
    debug_print("Testing LlamaVision Model Loader")
    llama_loader = LlamaVisionModelLoader()
    llama_models = llama_loader.INPUT_TYPES()['required']['model_name'][0]
    if llama_models:
        debug_print(f"Testing with first available LlamaVision model: {llama_models[0]}")
        llama_loader.load_model(llama_models[0])
    else:
        debug_print("No LlamaVision models found to test with")
    
    debug_print("Model loader test complete")
