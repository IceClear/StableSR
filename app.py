"""
This file is used for deploying hugging face demo:
https://huggingface.co/spaces/Iceclear/StableSR/
"""

import sys
sys.path.append('StableSR')
import os
import cv2
import torch
import torch.nn.functional as F
import gradio as gr
import torchvision
from torchvision.transforms.functional import normalize
from ldm.util import instantiate_from_config
from torch import autocast
import PIL
import numpy as np
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from omegaconf import OmegaConf
from PIL import Image
import copy
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from scripts.util_image import ImageSpliterTh
from basicsr.utils.download_util import load_file_from_url
from einops import rearrange, repeat
from pathlib import Path

# os.system("pip freeze")
ckpt_dir = Path('./weights')
if not ckpt_dir.exists():
	ckpt_dir.mkdir()

pretrain_model_url = {
	'stablesr_512': 'https://huggingface.co/Iceclear/StableSR/resolve/main/stablesr_000117.ckpt',
	'stablesr_768': 'https://huggingface.co/Iceclear/StableSR/resolve/main/stablesr_768v_000139.ckpt',
	'CFW': 'https://huggingface.co/Iceclear/StableSR/resolve/main/vqgan_cfw_00011.ckpt',
}
# download weights
if not os.path.exists('./weights/stablesr_000117.ckpt'):
	load_file_from_url(url=pretrain_model_url['stablesr_512'], model_dir='./weights/', progress=True, file_name=None)
if not os.path.exists('./weights/stablesr_768v_000139.ckpt'):
	load_file_from_url(url=pretrain_model_url['stablesr_768'], model_dir='./weights/', progress=True, file_name=None)
if not os.path.exists('./weights/vqgan_cfw_00011.ckpt'):
	load_file_from_url(url=pretrain_model_url['CFW'], model_dir='./weights/', progress=True, file_name=None)

# download images
torch.hub.download_url_to_file(
	'https://raw.githubusercontent.com/zsyOAOA/ResShift/master/testdata/RealSet128/Lincoln.png',
	'01.png')
torch.hub.download_url_to_file(
	'https://raw.githubusercontent.com/zsyOAOA/ResShift/master/testdata/RealSet128/oldphoto6.png',
	'02.png')
torch.hub.download_url_to_file(
	'https://raw.githubusercontent.com/zsyOAOA/ResShift/master/testdata/RealSet128/comic2.png',
	'03.png')
torch.hub.download_url_to_file(
	'https://raw.githubusercontent.com/zsyOAOA/ResShift/master/testdata/RealSet128/OST_120.png',
	'04.png')
torch.hub.download_url_to_file(
	'https://raw.githubusercontent.com/zsyOAOA/ResShift/master/testdata/RealSet65/comic3.png',
	'05.png')

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
							process to divide up.
	:param section_counts: either a list of numbers, or a string containing
							 comma-separated numbers, indicating the step count
							 per section. As a special case, use "ddimN" where N
							 is a number of steps to use the striding from the
							 DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda")
vqgan_config = OmegaConf.load("./configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
vq_model = load_model_from_config(vqgan_config, './weights/vqgan_cfw_00011.ckpt')
vq_model = vq_model.to(device)

os.makedirs('output', exist_ok=True)

def inference(image, upscale, dec_w, seed, model_type, ddpm_steps, colorfix_type):
	"""Run a single prediction on the model"""
	precision_scope = autocast
	vq_model.decoder.fusion_w = dec_w
	seed_everything(seed)

	if model_type == '512':
		config = OmegaConf.load("./configs/stableSRNew/v2-finetune_text_T_512.yaml")
		model = load_model_from_config(config, "./weights/stablesr_000117.ckpt")
		min_size = 512
	else:
		config = OmegaConf.load("./configs/stableSRNew/v2-finetune_text_T_768v.yaml")
		model = load_model_from_config(config, "./weights/stablesr_768v_000139.ckpt")
		min_size = 768

	model = model.to(device)
	model.configs = config
	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
							linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000

	sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	use_timesteps = set(space_timesteps(1000, [ddpm_steps]))
	last_alpha_cumprod = 1.0
	new_betas = []
	timestep_map = []
	for i, alpha_cumprod in enumerate(model.alphas_cumprod):
		if i in use_timesteps:
			new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
			last_alpha_cumprod = alpha_cumprod
			timestep_map.append(i)
	new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	model.num_timesteps = 1000
	model.ori_timesteps = list(use_timesteps)
	model.ori_timesteps.sort()
	model = model.to(device)

	try: # global try
			with torch.no_grad():
				with precision_scope("cuda"):
					with model.ema_scope():
						init_image = load_img(image)
						init_image = F.interpolate(
									init_image,
									size=(int(init_image.size(-2)*upscale),
											int(init_image.size(-1)*upscale)),
									mode='bicubic',
									)

						if init_image.size(-1) < min_size or init_image.size(-2) < min_size:
							ori_size = init_image.size()
							rescale = min_size * 1.0 / min(init_image.size(-2), init_image.size(-1))
							new_h = max(int(ori_size[-2]*rescale), min_size)
							new_w = max(int(ori_size[-1]*rescale), min_size)
							init_template = F.interpolate(
										init_image,
										size=(new_h, new_w),
										mode='bicubic',
										)
						else:
							init_template = init_image
							rescale = 1
						init_template = init_template.clamp(-1, 1)
						assert init_template.size(-1) >= min_size
						assert init_template.size(-2) >= min_size

						init_template = init_template.type(torch.float16).to(device)

						if init_template.size(-1) <= 1024 and init_template.size(-2) <= 1024:
							init_latent_generator, enc_fea_lq = vq_model.encode(init_template)
							init_latent = model.get_first_stage_encoding(init_latent_generator)
							text_init = ['']*init_template.size(0)
							semantic_c = model.cond_stage_model(text_init)

							noise = torch.randn_like(init_latent)

							t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
							t = t.to(device).long()
							x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)

							if init_template.size(-1)<= min_size and init_template.size(-2) <= min_size:
								samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, batch_size=init_template.size(0), timesteps=ddpm_steps, time_replace=ddpm_steps, x_T=x_T, return_intermediates=True)
							else:
								samples, _ = model.sample_canvas(cond=semantic_c, struct_cond=init_latent, batch_size=init_template.size(0), timesteps=ddpm_steps, time_replace=ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(min_size/8), tile_overlap=min_size//16, batch_size_sample=init_template.size(0))
							x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
							if colorfix_type == 'adain':
								x_samples = adaptive_instance_normalization(x_samples, init_template)
							elif colorfix_type == 'wavelet':
								x_samples = wavelet_reconstruction(x_samples, init_template)
							x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
						else:
							im_spliter = ImageSpliterTh(init_template, min(init_template.size(-1), init_template.size(-2), 1024), min(init_template.size(-1)-200, init_template.size(-2)-200, 768), sf=1)
							for im_lq_pch, index_infos in im_spliter:
								init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))  # move to latent space
								text_init = ['']*init_latent.size(0)
								semantic_c = model.cond_stage_model(text_init)
								noise = torch.randn_like(init_latent)
								# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
								t = repeat(torch.tensor([999]), '1 -> b', b=init_template.size(0))
								t = t.to(device).long()
								x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
								# x_T = noise
								samples, _ = model.sample_canvas(cond=semantic_c, struct_cond=init_latent, batch_size=im_lq_pch.size(0), timesteps=ddpm_steps, time_replace=ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(min_size/8), tile_overlap=min_size//16, batch_size_sample=im_lq_pch.size(0))
								_, enc_fea_lq = vq_model.encode(im_lq_pch)
								x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
								if colorfix_type == 'adain':
									x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
								elif colorfix_type == 'wavelet':
									x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
								im_spliter.update(x_samples, index_infos)
							x_samples = im_spliter.gather()
							x_samples = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)

			if rescale > 1:
				x_samples = F.interpolate(
							x_samples,
							size=(int(init_image.size(-2)),
									int(init_image.size(-1))),
							mode='bicubic',
							)
				x_samples = x_samples.clamp(0, 1)
			x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
			restored_img = x_sample.astype(np.uint8)
			Image.fromarray(x_sample.astype(np.uint8)).save(f'output/out.png')

			return restored_img, f'output/out.png'
	except Exception as error:
		print('Global exception', error)
		return None, None


title = "Exploiting Diffusion Prior for Real-World Image Super-Resolution"
description = r"""<center><img src='https://user-images.githubusercontent.com/22350795/236680126-0b1cdd62-d6fc-4620-b998-75ed6c31bf6f.png' style='height:40px' alt='StableSR logo'></center>
<b>Official Gradio demo</b> for <a href='https://github.com/IceClear/StableSR' target='_blank'><b>Exploiting Diffusion Prior for Real-World Image Super-Resolution</b></a>.<br>
🔥 StableSR is a general image super-resolution algorithm for real-world and AIGC images.<br>
"""
article = r"""
If StableSR is helpful, please help to ⭐ the <a href='https://github.com/IceClear/StableSR' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/IceClear/StableSR?style=social)](https://github.com/IceClear/StableSR)

---

📝 **Citation**

If our work is useful for your research, please consider citing:
```bibtex
@inproceedings{wang2023exploiting,
	author = {Wang, Jianyi and Yue, Zongsheng and Zhou, Shangchen and Chan, Kelvin CK and Loy, Chen Change},
	title = {Exploiting Diffusion Prior for Real-World Image Super-Resolution},
	booktitle = {arXiv preprint arXiv:2305.07015},
	year = {2023}
}
```

📋 **License**

This project is licensed under <a rel="license" href="https://github.com/IceClear/StableSR/blob/main/LICENSE.txt">S-Lab License 1.0</a>.
Redistribution and use for non-commercial purposes should follow this license.

📧 **Contact**

If you have any questions, please feel free to reach me out at <b>iceclearwjy@gmail.com</b>.

<div>
	🤗 Find Me:
	<a href="https://twitter.com/Iceclearwjy"><img style="margin-top:0.5em; margin-bottom:0.5em" src="https://img.shields.io/twitter/follow/Iceclearwjy?label=%40Iceclearwjy&style=social" alt="Twitter Follow"></a>
	<a href="https://github.com/IceClear"><img style="margin-top:0.5em; margin-bottom:2em" src="https://img.shields.io/github/followers/IceClear?style=social" alt="Github Follow"></a>
</div>

<center><img src='https://visitor-badge.laobi.icu/badge?page_id=IceClear/StableSR' alt='visitors'></center>
"""

demo = gr.Interface(
	inference, [
		gr.inputs.Image(type="filepath", label="Input"),
		gr.inputs.Number(default=1, label="Rescaling_Factor (Large images require huge time)"),
		gr.Slider(0, 1, value=0.5, step=0.01, label='CFW_Fidelity (0 for better quality, 1 for better identity)'),
		gr.inputs.Number(default=42, label="Seeds"),
		gr.Dropdown(
			choices=["512", "768v"],
			value="512",
			label="Model",
			),
		gr.Slider(10, 1000, value=200, step=1, label='Sampling timesteps for DDPM (Large steps for better quality, but huge time)'),
		gr.Dropdown(
			choices=["none", "adain", "wavelet"],
			value="adain",
			label="Color_Correction",
			),
	], [
		gr.outputs.Image(type="numpy", label="Output"),
		gr.outputs.File(label="Download the output")
	],
	title=title,
	description=description,
	article=article,
	examples=[
		['./01.png', 4, 0.5, 42, "512", 200, "adain"],
		['./02.png', 4, 0.5, 42, "512", 200, "adain"],
		['./03.png', 4, 0.5, 42, "512", 200, "adain"],
		['./04.png', 4, 0.5, 42, "512", 200, "adain"],
		['./05.png', 4, 0.5, 42, "512", 200, "adain"]
		]
	)

demo.queue(concurrency_count=1)
demo.launch()
