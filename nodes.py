import comfy.utils
import comfy.model_management as mm
import folder_paths
# Requires transformers >= 4.45.0
from transformers import LlavaForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, set_seed
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import time
import os
from pathlib import Path
import re

pixtral_model_dir = os.path.join(folder_paths.models_dir, "pixtral")
llama_vision_model_dir = os.path.join(folder_paths.models_dir, "llama-vision")
# Add pixtral and llama-vision folders if not present
if not os.path.exists(pixtral_model_dir):
    os.makedirs(pixtral_model_dir)
if not os.path.exists(llama_vision_model_dir):
    os.makedirs(llama_vision_model_dir)

class PixtralModelLoader:
    """Loads a Pixtral model. Add models as folders inside the `ComfyUI/models/pixtral` folder. Each model folder should contain a standard transformers loadable safetensors model along with a tokenizer and any config files needed."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([item.name for item in Path(pixtral_model_dir).iterdir() if item.is_dir()],),
            }
        }

    RETURN_TYPES = ("PIXTRAL_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "PixtralLlamaVision/Pixtral"
    TITLE = "Load Pixtral Model"

    def load_model(self, model_name):
        model_path = os.path.join(pixtral_model_dir, model_name)
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
    """Generates text using a Pixtral model. Takes a list of images and a string prompt as input. The prompt must contain an equal number of [IMG] tokens to the number of images passed in."""
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
        device = pixtral_model['model'].device
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
        output_tokens = generate_ids[0] if include_prompt_in_output else generate_ids[0][prompt_tokens:]
        output = pixtral_model['processor'].decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output)
        return (output,)


class LlamaVisionModelLoader:
    """Loads a Llama 3.2 Vision model. Add models as folders inside the `ComfyUI/models/llama-vision` folder. Each model folder should contain a standard transformers loadable safetensors model along with a tokenizer and any config files needed."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([item.name for item in Path(llama_vision_model_dir).iterdir() if item.is_dir()],),
            }
        }

    RETURN_TYPES = ("LLAMA_VISION_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "PixtralLlamaVision/LlamaVision"
    TITLE = "Load Llama Vision Model"

    def load_model(self, model_name):
        model_path = os.path.join(llama_vision_model_dir, model_name)
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
    """Generates text using a Llama 3.2 Vision model. The prompt must contain an equal number of <|image|> tokens to the number of images passed in. Image tokens must also be sequential and before the text you want them to apply to for the image attention to work as intended."""
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
        device = llama_vision_model['model'].device
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
        output_tokens = generate_ids[0] if include_prompt_in_output else generate_ids[0][prompt_tokens:]
        output = llama_vision_model['processor'].decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output)
        return (output,)


# Utility for bounding boxes (I'm sure this has been done before but I just wanted to try it out to see how well Pixtral can do it)
class ParseBoundingBoxes:
    """Uses a regular expression to find bounding boxes in a string, returning a list of bbox objects (compatible with mtb). `relative` means the bounding box uses float values between 0 and 1 if true and absolute image coordinates if false. `corners_only` means the bounding box is [(x1, y1), (x2, y2)] if true and [(x1, y1), (width, height)] if false. Parentheses are treated as optional."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "string": ("STRING",),
                "relative": ("BOOLEAN", {"default": True}),
                "corners_only": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "generate_bboxes"
    CATEGORY = "PixtralLlamaVision/Utility"
    TITLE = "Parse Bounding Boxes"

    def generate_bboxes(self, image, string, relative, corners_only):
        image_width = image.shape[2]
        image_height = image.shape[1]

        bboxes = []
        # Ridiculous-looking regex
        for match in re.findall(r"""\[?\(?([0-9\.]+),\s*([0-9\.]+)\)?,\s*\(?([0-9\.]+),\s*([0-9\.]+)\)?\]?""", string, flags=re.M):
            try:
                x1_raw = float(match[0])
                y1_raw = float(match[1])
                x2_raw = float(match[2])
                y2_raw = float(match[3])

                if relative:
                    x1 = int(image_width*x1_raw)
                    y1 = int(image_height*y1_raw)
                    x2 = int(image_width*x2_raw)
                    y2 = int(image_height*y2_raw)
                else:
                    x1 = int(x1_raw)
                    y1 = int(y1_raw)
                    x2 = int(x2_raw)
                    y2 = int(y2_raw)

                if corners_only:
                    width = x2 - x1
                    height = y2 - y1
                else:
                    width = x2
                    height = y2
                
                if width <= 0 or width > image_width or height <= 0 or height > image_height:
                    print(f"Invalid bbox: ({x1}, {y1}, {width}, {height})")
                    continue
                bbox = (x1, y1, width, height)
                bboxes.append(bbox)
            except Exception as e:
                print(f"Failed to parse bbox: {match}")

        return (bboxes,)


def process_regex_flags(flags):
    flag_value = re.NOFLAG
    if 'a' in flags.lower():
        flag_value |= re.A
    if 'i' in flags.lower():
        flag_value |= re.I
    if 'l' in flags.lower():
        flag_value |= re.L
    if 'm' in flags.lower():
        flag_value |= re.M
    if 's' in flags.lower():
        flag_value |= re.S
    if 'u' in flags.lower(): # u for useless
        flag_value |= re.U
    if 'x' in flags.lower():
        flag_value |= re.X
    return flag_value

# Utility nodes that I couldn't find elsewhere, not sure why?
class RegexSplitString:
    """Uses a regular expression to split in a string by a pattern into a list of strings"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pattern": ("STRING",),
                "string": ("STRING",),
                "flags": ("STRING", {"default": "M"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "split_string"
    CATEGORY = "PixtralLlamaVision/Utility"
    TITLE = "Regex Split String"

    def split_string(self, pattern, string, flags):
        return (re.split(pattern, string, flags=process_regex_flags(flags)),)


class RegexSearch:
    """Uses a regular expression to search for the first occurrence of a pattern in a string, returning whether the pattern was found, the start and end positions if found, and the list of match groups if found"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pattern": ("STRING",),
                "string": ("STRING",),
                "flags": ("STRING", {"default": "M"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "INT", "STRING")
    FUNCTION = "search"
    CATEGORY = "PixtralLlamaVision/Utility"
    TITLE = "Regex Search"

    def search(self, pattern, string, flags):
        match = re.search(pattern, string, flags=process_regex_flags(flags))
        if match:
            span = match.span()
            groups = list(match.groups())
            return (True, span[0], span[1], groups)
        return (False, 0, 0, [])


class RegexFindAll:
    """Uses a regular expression to find all matches of a pattern in a string, returning a list of match groups (which could be strings or tuples of strings if you have more than one match group)"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pattern": ("STRING",),
                "string": ("STRING",),
                "flags": ("STRING", {"default": "M"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "find_all"
    CATEGORY = "PixtralLlamaVision/Utility"
    TITLE = "Regex Find All"

    def find_all(self, pattern, string, flags):
        return (re.findall(pattern, string, flags=process_regex_flags(flags)),)


# This one is also available in Derfuu_ComfyUI_ModdedNodes
class RegexSubstitution:
    """Uses a regular expression to find and replace text in a string"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pattern": ("STRING",),
                "string": ("STRING",),
                "replace": ("STRING",),
                "flags": ("STRING", {"default": "M"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "sub"
    CATEGORY = "PixtralLlamaVision/Utility"
    TITLE = "Regex Substitution"

    def sub(self, pattern, string, replace, flags):
        return (re.sub(pattern, replace, string, flags=process_regex_flags(flags)),)


class JoinString:
    """Joins a list of strings with a delimiter between them"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string_list": ("STRING",),
                "delimiter": ("STRING", {"default": ", "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "join_string"
    CATEGORY = "PixtralLlamaVision/Utility"
    TITLE = "Join String"

    def join_string(self, string_list, delimiter):
        # Convert to strings just in case? Or is this a bad idea? Well, it'll error if they're not strings, so I guess this will have to do
        return (delimiter.join([str(string) for string in string_list]),)


# Arbitrary data type for list/tuple indexing
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

ANY = AnyType("*")

# These ones are especially weird to not be doable in ComfyUI base
class SelectIndex:
    """Returns list[index]"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list": (ANY,),
                "index": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = (ANY,)
    FUNCTION = "select_index"
    CATEGORY = "PixtralLlamaVision/Utility"
    TITLE = "Select Index"

    def select_index(self, list, index):
        return (list[index],)

class SliceList:
    """Returns list[start_index:end_index]"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list": (ANY,),
                "start_index": ("INT", {"default": 0}),
                "end_index": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = (ANY,)
    FUNCTION = "select_index"
    CATEGORY = "PixtralLlamaVision/Utility"
    TITLE = "Slice List"

    def select_index(self, list, start_index, end_index):
        return (list[start_index:end_index],)

# Batch Count works for getting list length

NODE_CLASS_MAPPINGS = {
    "PixtralModelLoader": PixtralModelLoader,
    "PixtralGenerateText": PixtralGenerateText,
    # Not really much need to work with the image tokenization directly for something like image captioning, but might be interesting later...
    #"PixtralImageEncode": PixtralImageEncode,
    #"PixtralTextEncode": PixtralTextEncode,
    "LlamaVisionModelLoader": LlamaVisionModelLoader,
    "LlamaVisionGenerateText": LlamaVisionGenerateText,
    "RegexSplitString": RegexSplitString,
    "RegexSearch": RegexSearch,
    "RegexFindAll": RegexFindAll,
    "RegexSubstitution": RegexSubstitution,
    "JoinString": JoinString,
    "ParseBoundingBoxes": ParseBoundingBoxes,
    "SelectIndex": SelectIndex,
    "SliceList": SliceList,
}

NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}