# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import PIL
import numpy as np
import copy
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from pytorch_lightning import seed_everything
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from scripts.wavelet_color_fix import (
    wavelet_reconstruction,
    adaptive_instance_normalization,
)

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        config = OmegaConf.load("configs/stableSRNew/v2-finetune_text_T_512.yaml")
        self.model = load_model_from_config(config, "stablesr_000117.ckpt")
        device = torch.device("cuda")

        self.model.configs = config
        self.model = self.model.to(device)

        vqgan_config = OmegaConf.load(
            "configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml"
        )
        self.vq_model = load_model_from_config(vqgan_config, "vqgan_cfw_00011.ckpt")
        self.vq_model = self.vq_model.to(device)

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        ddpm_steps: int = Input(
            description="Number of DDPM steps for sampling", default=200
        ),
        fidelity_weight: float = Input(
            description="Balance the quality (lower number) and fidelity (higher number)",
            default=0.5,
        ),
        upscale: float = Input(
            description="The upscale for super-resolution, 4x SR by default",
            default=4.0,
        ),
        tile_overlap: int = Input(
            description="The overlap between tiles, betwwen 0 to 64",
            ge=0,
            le=64,
            default=32,
        ),
        colorfix_type: str = Input(
            choices=["adain", "wavelet", "none"], default="adain"
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        self.vq_model.decoder.fusion_w = fidelity_weight

        seed_everything(seed)

        n_samples = 1
        device = torch.device("cuda")

        cur_image = load_img(str(input_image)).to(device)
        cur_image = F.interpolate(
            cur_image,
            size=(int(cur_image.size(-2) * upscale), int(cur_image.size(-1) * upscale)),
            mode="bicubic",
        )

        self.model.register_schedule(
            given_betas=None,
            beta_schedule="linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.0120,
            cosine_s=8e-3,
        )
        self.model.num_timesteps = 1000

        sqrt_alphas_cumprod = copy.deepcopy(self.model.sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = copy.deepcopy(
            self.model.sqrt_one_minus_alphas_cumprod
        )

        use_timesteps = set(space_timesteps(1000, [ddpm_steps]))
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(self.model.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        new_betas = [beta.data.cpu().numpy() for beta in new_betas]
        self.model.register_schedule(
            given_betas=np.array(new_betas), timesteps=len(new_betas)
        )
        self.model.num_timesteps = 1000
        self.model.ori_timesteps = list(use_timesteps)
        self.model.ori_timesteps.sort()
        self.model = self.model.to(device)

        precision_scope = autocast
        input_size = 512

        output = "/tmp/out.png"

        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    init_image = cur_image
                    init_image = init_image.clamp(-1.0, 1.0)
                    ori_size = None

                    print(init_image.size())

                    if (
                        init_image.size(-1) < input_size
                        or init_image.size(-2) < input_size
                    ):
                        ori_size = init_image.size()
                        new_h = max(ori_size[-2], input_size)
                        new_w = max(ori_size[-1], input_size)
                        init_template = torch.zeros(
                            1, init_image.size(1), new_h, new_w
                        ).to(init_image.device)
                        init_template[:, :, : ori_size[-2], : ori_size[-1]] = init_image
                    else:
                        init_template = init_image

                    init_latent = self.model.get_first_stage_encoding(
                        self.model.encode_first_stage(init_template)
                    )  # move to latent space
                    text_init = [""] * n_samples
                    semantic_c = self.model.cond_stage_model(text_init)

                    noise = torch.randn_like(init_latent)
                    # If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
                    t = repeat(torch.tensor([999]), "1 -> b", b=init_image.size(0))
                    t = t.to(device).long()
                    x_T = self.model.q_sample_respace(
                        x_start=init_latent,
                        t=t,
                        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                        noise=noise,
                    )
                    samples, _ = self.model.sample_canvas(
                        cond=semantic_c,
                        struct_cond=init_latent,
                        batch_size=init_image.size(0),
                        timesteps=ddpm_steps,
                        time_replace=ddpm_steps,
                        x_T=x_T,
                        return_intermediates=True,
                        tile_size=int(input_size / 8),
                        tile_overlap=tile_overlap,
                        batch_size_sample=n_samples,
                    )
                    _, enc_fea_lq = self.vq_model.encode(init_template)
                    x_samples = self.vq_model.decode(
                        samples * 1.0 / self.model.scale_factor, enc_fea_lq
                    )
                    if ori_size is not None:
                        x_samples = x_samples[:, :, : ori_size[-2], : ori_size[-1]]
                    if colorfix_type == "adain":
                        x_samples = adaptive_instance_normalization(
                            x_samples, init_image
                        )
                    elif colorfix_type == "wavelet":
                        x_samples = wavelet_reconstruction(x_samples, init_image)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for i in range(init_image.size(0)):
                        x_sample = 255.0 * rearrange(
                            x_samples[i].cpu().numpy(), "c h w -> h w c"
                        )
                        Image.fromarray(x_sample.astype(np.uint8)).save(output)

        return Path(output)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def read_image(im_path):
    im = np.array(Image.open(im_path).convert("RGB"))
    im = im.astype(np.float32) / 255.0
    im = im[None].transpose(0, 3, 1, 2)
    im = (torch.from_numpy(im) - 0.5) / 0.5

    return im.cuda()


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]  # [250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0
