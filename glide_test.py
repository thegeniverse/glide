import torchvision
from PIL import Image

from glide.modeling_utils import load_model

try:

    glide_model = load_model(timestep_respacing=100)

    prompt = "A dog with the face of a goose"
    batch_size = 2

    img = torchvision.transforms.PILToTensor()(
        Image.open("./img.jpg"))[None, :]
    img_mask = torchvision.transforms.PILToTensor()(
        Image.open("./img_mask.jpg"))[None, :]

    gen_img_list = glide_model.inpaint(
        prompt=prompt,
        img=img,
        img_mask=img_mask,
        batch_size=batch_size,
    )

    for gen_idx, gen_img in enumerate(gen_img_list):
        gen_img.save(f"inpaint_{gen_idx}.png")

    gen_img_list = glide_model.text2im(
        prompt=prompt,
        batch_size=batch_size,
    )

    for gen_idx, gen_img in enumerate(gen_img_list):
        gen_img.save(f"text2im_{gen_idx}.png")

    gen_img_list = glide_model.clip_guided(
        prompt=prompt,
        batch_size=batch_size,
    )

    for gen_idx, gen_img in enumerate(gen_img_list):
        gen_img.save(f"clip_guided_{gen_idx}.png")

    print("OK!")

except Exception as e:
    print("ERROR!")
    print(e)