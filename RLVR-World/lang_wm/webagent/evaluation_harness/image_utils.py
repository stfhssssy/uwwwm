# This file is from WMA project:
# https://github.com/kyle8581/WMA-Agents

from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)
import torch


def get_captioning_fn(
    device: str | torch.device | int,
    dtype: torch.dtype,
    model_name: str = "Salesforce/blip2-flan-t5-xl"
) -> Callable[[Sequence[Image.Image], Optional[list[str]], int], list[str]]:
    if "blip2" in model_name:
        captioning_processor = Blip2Processor.from_pretrained(model_name)
        captioning_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
    else:
        raise NotImplementedError(
            "Only BLIP-2 models are currently supported"
        )
    captioning_model.to(device)

    def caption_images(
        images: Sequence[Image.Image],
        prompt: Optional[list[str]] = None,
        max_new_tokens: int = 32,
    ) -> list[str]:
        if prompt is None:
            # Perform VQA
            inputs = captioning_processor(
                images=images, return_tensors="pt"
            ).to(device, dtype)
            generated_ids = captioning_model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            captions = captioning_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            # Regular captioning. Prompt is a list of strings, one for each image
            assert len(images) == len(
                prompt
            ), "Number of images and prompts must match, got {} and {}".format(
                len(images), len(prompt)
            )
            inputs = captioning_processor(
                images=images, text=prompt, return_tensors="pt"
            ).to(device, dtype)
            generated_ids = captioning_model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            captions = captioning_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        return captions

    return caption_images


def get_image_ssim(imageA: Image.Image, imageB: Image.Image) -> float:
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(
        imageA.size[1], imageB.size[1]
    )

    # Resize images
    imageA = imageA.resize(new_size, Image.Resampling.LANCZOS)
    imageB = imageB.resize(new_size, Image.Resampling.LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA_ = np.array(grayA)
    grayB_ = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA_, grayB_, full=True)
    return score
