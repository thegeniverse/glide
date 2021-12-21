from typing import *

import torch
import torchvision
from PIL import Image

from glide.clip.model_creation import create_clip_model
from glide.download import load_checkpoint
from glide.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide.tokenizer.simple_tokenizer import SimpleTokenizer


class GlideModel:
    def __init__(
        self,
        diff_config_dict: Dict = None,
        up_config_dict: Dict = None,
        device: str = "cuda",
    ):
        self.diff_config_dict = diff_config_dict
        self.up_config_dict = up_config_dict
        self.device = device

        self.model, self.diffusion = create_model_and_diffusion(
            **self.diff_config_dict, )
        self.model.eval()

        if self.device == "cuda":
            self.model.convert_to_fp16()

        self.model.to(self.device)
        self.model.load_state_dict(load_checkpoint("base", self.device))

        print("Diffusion model total base parameters",
              sum(x.numel() for x in self.model.parameters()))

        self.up_model, self.up_diffusion = create_model_and_diffusion(
            **self.up_config_dict, )
        self.up_model.eval()

        if self.device == "cuda":
            self.up_model.convert_to_fp16()

        self.up_model.to(self.device)
        self.up_model.load_state_dict(
            load_checkpoint(
                "upsample",
                self.device,
            ))
        print("upsampler model total parameters",
              sum(x.numel() for x in self.up_model.parameters()))

        self.clip_model = create_clip_model(device=self.device, )
        self.clip_model.image_encoder.load_state_dict(
            load_checkpoint(
                "clip/image-enc",
                self.device,
            ))
        self.clip_model.text_encoder.load_state_dict(
            load_checkpoint(
                "clip/text-enc",
                self.device,
            ))


def load_model(
    diff_config_dict: Dict = None,
    up_config_dict: Dict = None,
    device: str = "cuda",
    timestep_respacing: int = 100,
):
    if diff_config_dict is None:
        diff_config_dict = model_and_diffusion_defaults()
        diff_config_dict["use_fp16"] = device == "cuda"
        diff_config_dict[
            "timestep_respacing"] = f"{timestep_respacing}"  # use 100 diffusion steps for fast sampling

    if up_config_dict is None:
        up_config_dict = model_and_diffusion_defaults_upsampler()
        up_config_dict["use_fp16"] = True if device == "cuda" else False
        up_config_dict[
            "timestep_respacing"] = "fast27"  # use 27 diffusion steps for very fast sampling

    glide_model = GlideModel(
        diff_config_dict,
        up_config_dict,
        device,
    )

    return glide_model