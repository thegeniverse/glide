from typing import *

import torch
import torchvision

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

        self.model = None
        self.diffusion = None
        self.up_model = None
        self.up_difussion = None

        self.clip_model = None

        self.inpaint_model = None
        self.inpaint_diffusion = None
        self.up_inpaint_model = None
        self.up_inpaint_diffusion = None

    def load_base_models(self, ):
        self.model, self.diffusion = create_model_and_diffusion(
            **self.diff_config_dict, )
        self.model.eval()

        if "cuda" in self.device:
            self.model.convert_to_fp16()

        self.model.to(self.device)
        self.model.load_state_dict(load_checkpoint("base", self.device))

        print("Glide model total base parameters",
              sum(x.numel() for x in self.model.parameters()))

        self.up_model, self.up_diffusion = create_model_and_diffusion(
            **self.up_config_dict, )
        self.up_model.eval()

        if "cuda" in self.device:
            self.up_model.convert_to_fp16()

        self.up_model.to(self.device)
        self.up_model.load_state_dict(
            load_checkpoint(
                "upsample",
                self.device,
            ))
        print("upsampler model total parameters",
              sum(x.numel() for x in self.up_model.parameters()))

    def load_clip_model(self, ):
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

    def load_inpaint_models(self, ):
        inpaint_config_dict = self.diff_config_dict
        inpaint_config_dict["inpaint"] = True

        inpaint_up_config_dict = self.up_config_dict
        inpaint_up_config_dict["inpaint"] = True

        self.inpaint_model, self.inpaint_diffusion = create_model_and_diffusion(
            **inpaint_config_dict, )
        self.inpaint_model.eval()

        if "cuda" in self.device:
            self.inpaint_model.convert_to_fp16()

        self.inpaint_model.to(self.device)
        self.inpaint_model.load_state_dict(
            load_checkpoint("base-inpaint", self.device))

        self.up_inpaint_model, self.up_inpaint_diffusion = create_model_and_diffusion(
            **inpaint_up_config_dict)
        self.up_inpaint_model.eval()

        if "cuda" in self.device:
            self.up_inpaint_model.convert_to_fp16()

        self.up_inpaint_model.to(self.device)
        self.up_inpaint_model.load_state_dict(
            load_checkpoint('upsample-inpaint', self.device))

    def text2im(
        self,
        prompt: str,
        batch_size: int = 1,
        width: int = 64,
        height: int = 64,
        upsample_scale: int = 4,
        upsample_temp: float = 0.997,
        guidance_scale: float = 3.0,
    ):
        full_batch_size = batch_size * 2

        if self.model is None:
            self.load_base_models()

        tokens = self.model.tokenizer.encode(prompt)
        tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
            tokens, self.diff_config_dict['text_ctx'])

        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
            [], self.diff_config_dict['text_ctx'])

        model_kwargs = dict(
            tokens=torch.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size,
                device=self.device,
            ),
            mask=torch.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
        )

        def model_fn(x_t, ts, **kwargs):
            half = x_t[:len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        samples = self.diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, height, width),  # only thing that's changed
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.model.del_cache()

        tokens = self.up_model.tokenizer.encode(prompt)
        tokens, mask = self.up_model.tokenizer.padded_tokens_and_mask(
            tokens, self.up_config_dict['text_ctx'])

        model_kwargs = dict(
            low_res=((samples + 1) * 127.5).round() / 127.5 - 1,
            tokens=torch.tensor([tokens] * batch_size, device=self.device),
            mask=torch.tensor(
                [mask] * batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
        )

        up_shape = (
            batch_size,
            3,
            height * upsample_scale,
            width * upsample_scale,
        )

        up_samples = self.up_diffusion.ddim_sample_loop(
            self.up_model,
            up_shape,
            noise=torch.randn(up_shape, device=self.device) * upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.up_model.del_cache()

        gen_img_list = []

        for gen_tensor_img in up_samples:
            gen_img_list.append(
                torchvision.transforms.ToPILImage(mode="RGB")(
                    (gen_tensor_img + 1) / 2))

        return gen_img_list

    def inpaint(
        self,
        prompt: str,
        img: torch.Tensor,
        img_mask: torch.Tensor,
        batch_size: int = 1,
        guidance_scale: float = 5.0,
        upsample_scale: int = 4,
        upsample_temp: float = 0.997,
    ):
        full_batch_size = batch_size * 2

        if self.inpaint_model is None:
            self.load_inpaint_models()

        if self.clip_model is None:
            self.load_clip_model()

        img = torch.nn.functional.interpolate(
            img,
            (256, 256),
            mode="nearest",
        ).to(self.device, torch.float32)
        img_mask = torch.nn.functional.interpolate(
            img_mask,
            (256, 256),
            mode="nearest",
        ).to(self.device, torch.float32)

        if img_mask.shape[1] > 1:
            img_mask = img_mask[:, 0, ::][:, None, ::]

        if img.max() > 1:
            img /= 255.
            img = img * 2 - 1

        if img_mask.max() > 1:
            img_mask /= 255.

        height, width = (64, 64)

        img_64 = torch.nn.functional.interpolate(
            img,
            (64, 64),
            mode="nearest",
        )
        img_mask_64 = torch.nn.functional.interpolate(
            img_mask,
            (64, 64),
            mode="nearest",
        )

        tokens = self.inpaint_model.tokenizer.encode(prompt)
        tokens, mask = self.inpaint_model.tokenizer.padded_tokens_and_mask(
            tokens, self.diff_config_dict['text_ctx'])

        uncond_tokens, uncond_mask = self.inpaint_model.tokenizer.padded_tokens_and_mask(
            [], self.diff_config_dict['text_ctx'])

        model_kwargs = dict(
            tokens=torch.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size,
                device=self.device,
            ),
            mask=torch.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
            inpaint_image=(img_64 * img_mask_64).repeat(
                full_batch_size, 1, 1, 1).to(self.device),
            inpaint_mask=img_mask_64.repeat(full_batch_size, 1, 1,
                                            1).to(self.device),
        )

        def model_fn(x_t, ts, **kwargs):
            half = x_t[:len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.inpaint_model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        def denoised_fn(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                x_start * (1 - model_kwargs['inpaint_mask']) +
                model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask'])

        samples = self.inpaint_diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, height, width),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            denoised_fn=denoised_fn,
        )[:batch_size]
        self.inpaint_model.del_cache()

        tokens = self.up_inpaint_model.tokenizer.encode(prompt)
        tokens, mask = self.up_inpaint_model.tokenizer.padded_tokens_and_mask(
            tokens, self.up_config_dict['text_ctx'])

        model_kwargs = dict(
            low_res=((samples + 1) * 127.5).round() / 127.5 - 1,
            tokens=torch.tensor([tokens] * batch_size, device=self.device),
            mask=torch.tensor(
                [mask] * batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
            # Masked inpainting image.
            inpaint_image=(img * img_mask).repeat(batch_size, 1, 1,
                                                  1).to(self.device),
            inpaint_mask=img_mask.repeat(batch_size, 1, 1, 1).to(self.device),
        )

        up_shape = (
            batch_size,
            3,
            height * upsample_scale,
            width * upsample_scale,
        )

        def denoised_fn(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                x_start * (1 - model_kwargs['inpaint_mask']) +
                model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask'])

        up_samples = self.up_inpaint_diffusion.ddim_sample_loop(
            self.up_inpaint_model,
            up_shape,
            noise=torch.randn(up_shape, device=self.device) * upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            denoised_fn=denoised_fn,
        )[:batch_size]
        self.up_inpaint_model.del_cache()

        gen_img_list = []

        for gen_tensor_img in up_samples:
            gen_img_list.append(
                torchvision.transforms.ToPILImage(mode="RGB")(
                    (gen_tensor_img + 1) / 2))

        return gen_img_list

    def clip_guided(
        self,
        prompt: str,
        batch_size: int = 1,
        upsample_scale: int = 4,
        upsample_temp: float = 0.997,
        guidance_scale: float = 3.0,
    ):
        if self.model is None:
            self.load_base_models()

        if self.clip_model is None:
            self.load_clip_model()

        tokens = self.model.tokenizer.encode(prompt)
        tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
            tokens, self.diff_config_dict['text_ctx'])

        model_kwargs = dict(
            tokens=torch.tensor([tokens] * batch_size, device=self.device),
            mask=torch.tensor([mask] * batch_size,
                              dtype=torch.bool,
                              device=self.device),
        )

        cond_fn = self.clip_model.cond_fn(
            [prompt] * batch_size,
            guidance_scale,
        )

        samples = self.diffusion.p_sample_loop(
            self.model,
            (batch_size, 3, self.diff_config_dict["image_size"],
             self.diff_config_dict["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
        )
        self.model.del_cache()

        tokens = self.up_model.tokenizer.encode(prompt)
        tokens, mask = self.up_model.tokenizer.padded_tokens_and_mask(
            tokens, self.up_config_dict['text_ctx'])

        model_kwargs = dict(
            low_res=((samples + 1) * 127.5).round() / 127.5 - 1,
            tokens=torch.tensor([tokens] * batch_size, device=self.device),
            mask=torch.tensor(
                [mask] * batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
        )

        up_shape = (
            batch_size,
            3,
            self.diff_config_dict["image_size"] * upsample_scale,
            self.diff_config_dict["image_size"] * upsample_scale,
        )

        up_samples = self.up_diffusion.ddim_sample_loop(
            self.up_model,
            up_shape,
            noise=torch.randn(up_shape, device=self.device) * upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.up_model.del_cache()

        gen_img_list = []

        for gen_tensor_img in up_samples:
            gen_img_list.append(
                torchvision.transforms.ToPILImage(mode="RGB")(
                    (gen_tensor_img + 1) / 2))

        return gen_img_list

    def generate_from_prompt(
        self,
        prompt: str,
        mode: str = "text2im",
        batch_size: int = 1,
        *args,
        **kwargs,
    ):
        if mode == "text2im":
            results = self.text2im(
                prompt=prompt,
                batch_size=batch_size,
                **kwargs,
            )

        elif mode == "inpaint":
            results = self.inpaint(
                prompt=prompt,
                batch_size=batch_size,
                **kwargs,
            )

        elif mode == "clip_guided":
            results = self.clip_guided(
                prompt=prompt,
                batch_size=batch_size,
                **kwargs,
            )


def load_model(
    device: str = "cuda",
    timestep_respacing: int = 100,
    diff_config_dict: Dict = None,
    up_config_dict: Dict = None,
):
    if diff_config_dict is None:
        diff_config_dict = model_and_diffusion_defaults()
        diff_config_dict["use_fp16"] = "cuda" in device
        diff_config_dict[
            "timestep_respacing"] = f"{timestep_respacing}"  # use 100 diffusion steps for fast sampling

    if up_config_dict is None:
        up_config_dict = model_and_diffusion_defaults_upsampler()
        up_config_dict["use_fp16"] = True if "cuda" in device else False
        up_config_dict[
            "timestep_respacing"] = "fast27"  # use 27 diffusion steps for very fast sampling

    glide_model = GlideModel(
        diff_config_dict,
        up_config_dict,
        device,
    )

    return glide_model


def download_model():
    glide_model = load_model()

    glide_model.load_base_models()
    glide_model.load_inpaint_models()
    glide_model.load_clip_model()

    del glide_model
    torch.cuda.empty_cache()