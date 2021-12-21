import torch
import torchvision
from glide.modeling_utils import load_model

try:
    glide_model = load_model()

    prompt = "the sunrise shining over guatemala"
    batch_size = 1
    guidance_scale = 3.0

    # HACK: Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    tokens = glide_model.model.tokenizer.encode(prompt)
    tokens, mask = glide_model.model.tokenizer.padded_tokens_and_mask(
        tokens, glide_model.diff_config_dict['text_ctx'])

    # Pack the tokens together into glide_model.model kwargs.
    model_kwargs = dict(
        tokens=torch.tensor([tokens] * batch_size, device=glide_model.device),
        mask=torch.tensor([mask] * batch_size,
                          dtype=torch.bool,
                          device=glide_model.device),
    )

    # Setup guidance function for CLIP glide_model.model.
    cond_fn = glide_model.clip_model.cond_fn([prompt] * batch_size,
                                             guidance_scale)

    # Sample from the base glide_model.model.
    glide_model.model.del_cache()
    samples = glide_model.diffusion.p_sample_loop(
        glide_model.model,
        (batch_size, 3, glide_model.diff_config_dict["image_size"],
         glide_model.diff_config_dict["image_size"]),
        device=glide_model.device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )
    glide_model.model.del_cache()

    # Show the output
    # show_images(samples)

    ##############################
    # Upsample the 64x64 samples #
    ##############################

    tokens = glide_model.up_model.tokenizer.encode(prompt)
    tokens, mask = glide_model.up_model.tokenizer.padded_tokens_and_mask(
        tokens, glide_model.up_config_dict['text_ctx'])

    # Create the glide_model.model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples + 1) * 127.5).round() / 127.5 - 1,

        # Text tokens
        tokens=torch.tensor([tokens] * batch_size, device=glide_model.device),
        mask=torch.tensor(
            [mask] * batch_size,
            dtype=torch.bool,
            device=glide_model.device,
        ),
    )

    # Sample from the base glide_model.model.
    glide_model.up_model.del_cache()
    up_shape = (batch_size, 3, glide_model.up_config_dict["image_size"],
                glide_model.up_config_dict["image_size"])
    up_samples = glide_model.up_diffusion.ddim_sample_loop(
        glide_model.up_model,
        up_shape,
        noise=torch.randn(up_shape, device=glide_model.device) * upsample_temp,
        device=glide_model.device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    glide_model.up_model.del_cache()

    torchvision.transforms.ToPILImage(mode="RGB")(
        (up_samples[0] + 1) / 2).save("result.png")

    # Show the output
    # show_images(up_samples)

    print("OK!")

except Exception as e:
    print("ERROR!")
    print(e)