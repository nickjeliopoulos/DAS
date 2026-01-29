from PIL import Image
import os
from pathlib import Path
import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from transformers import pipeline
from diffusers.utils import load_image, numpy_to_pil, pt_to_pil
from importlib import resources

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

try:
    import hpsv2
    from hpsv2.img_score import (
        initialize_model as hpsv2_initialize_model,
        model_dict as hpsv2_model_dict,
    )
    from hpsv2.utils import (
        hps_version_map as hpsv2_huggingface_hub_map,
    )

    from hpsv2.src.open_clip import (
        create_model_and_transforms as hpsv2_create_model_and_transforms,
        get_tokenizer as hpsv2_get_tokenizer,
        OPENAI_DATASET_MEAN,
        OPENAI_DATASET_STD,
    )

except ImportError:
    print(
        f"HPSv2 not able to be imported, see https://github.com/tgxs002/HPSv2?tab=readme-ov-file#image-comparison for install"
    )
    print(
        f"Please download the model from https://huggingface.co/tgxs002/HPSv2 and place it in the cache_dir/"
    )

try:
    import ImageReward
except ImportError:
    print(
        f"Imagereward not able to be imported, see https://github.com/THUDM/ImageReward/tree/main for install"
    )

ASSETS_PATH = resources.files("assets")

def jpeg_compressibility(inference_dtype=None, device=None):
    import io
    import numpy as np
    def loss_fn(images):
        if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        loss = torch.tensor(sizes, dtype=inference_dtype, device=device)
        rewards = -1 * loss

        return loss, rewards

    return loss_fn

def clip_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from das.scorers.clip_scorer import CLIPScorer

    scorer = CLIPScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn

def aesthetic_score(
    torch_dtype=None,
    aesthetic_target=None,
    grad_scale=0,
    device=None,
    return_loss=False,
):
    from das.scorers.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)

            if aesthetic_target is None: # default maximization
                loss = -1 * scores
            else:
                # using L1 to keep on same scale
                loss = abs(scores - aesthetic_target)
            return loss * grad_scale, scores

        return loss_fn


def hps_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from das.scorers.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = 1.0 - scores
            return loss, scores

        return loss_fn



def imagereward_grad_tensor_transform(size: int = 224):
    ### Used pt_to_pil and numpy_to_pil as reference for clamp(...) transform
    return Compose(
        [
            lambda x: ((x / 2) + 0.5).clamp(0, 1),
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            # CenterCrop(size),
            # lambda x: x.convert("RGB"),
            # ToTensor(),
            Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )

### Adapted from ImageReward.py to accomodate gradients
def imagereward_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False
    ):
    ### Load the model
    imagereward_model = ImageReward.load("ImageReward-v1.0", device=device)
    imagereward_model = imagereward_model.eval()
    imagereward_xform = imagereward_grad_tensor_transform(224)

    def fitness_fn(images, prompts):
        text_input = imagereward_model.blip.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        # print(f"Images dtype: {images.dtype}\nInference dtype: {inference_dtype}")
        
        img_tensor = imagereward_xform(images).to(torch.float32)
        rewards = imagereward_model.score_gard(
            prompt_attention_mask=text_input.attention_mask,
            prompt_ids=text_input.input_ids,
            image=img_tensor,
        )

        if len(rewards.shape) == 2 and rewards.shape[1] == 1:
            rewards = rewards.squeeze(1)

        return rewards if not return_loss else (-rewards, rewards)

    return fitness_fn

# def imagereward_score(
#     inference_dtype=None, 
#     device=None, 
#     return_loss=False, 
# ):
#     from das.scorers.ImageReward_scorer import ImageRewardScorer

#     scorer = ImageRewardScorer(dtype=torch.float32, device=device)
#     scorer.requires_grad_(False)

#     if not return_loss:
#         def _fn(images, prompts):
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)
#             scores = scorer(images, prompts)
#             return scores

#         return _fn

#     else:
#         def loss_fn(images, prompts):
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)
#             scores = scorer(images, prompts)

#             loss = - scores
#             return loss, scores

#         return loss_fn

# def handle_input(img, skip: bool = False) -> Image:
#     if skip:
#         return img
#     if isinstance(img, torch.Tensor):
#         pil_imgs = pt_to_pil(img)
#     elif isinstance(img, np.ndarray):
#         pil_imgs = numpy_to_pil(img)
#     else:
#         pil_imgs = img
#     return pil_imgs

# def imagereward_score(
#     inference_dtype=None, 
#     device=None, 
#     return_loss=False, 
# ):
#     ### Load the model
#     imagereward_model = ImageReward.load("ImageReward-v1.0", device=device)
#     imagereward_model = imagereward_model.eval()

#     if not return_loss:
#         def fitness_fn(images, prompts) -> float:
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)
#             rewards = imagereward_model.score(prompts, images)
#             return torch.Tensor([rewards])
#     else:
#         def fitness_fn(images, prompts) -> float:
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)

#             rewards = imagereward_model.score(prompts, images)
#             return -torch.Tensor([rewards]), torch.Tensor([rewards])

#     return fitness_fn


def PickScore(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from das.scorers.PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn