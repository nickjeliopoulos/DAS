from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from das.diffusers_patch.pipeline_using_SMC import pipeline_using_smc
from das.diffusers_patch.pipeline_using_SMC_SDXL import pipeline_using_smc_sdxl
from das.diffusers_patch.pipeline_using_SMC_LCM import pipeline_using_smc_lcm
import numpy as np
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from DiffusionSampler import DiffusionModelSampler
import matplotlib.pyplot as plt

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

class DAS(DiffusionModelSampler):

    def __init__(self, config):
        super().__init__(config)

        if "xl" in self.config.pretrained.model:
            print("Using SDXL")
            self.pipeline_using_smc = pipeline_using_smc_sdxl
        elif "lcm" in self.config.pretrained.model or "LCM" in self.config.pretrained.model:
            print("Using LCM")
            self.pipeline_using_smc = pipeline_using_smc_lcm
        else:
            print("Using SD")
            self.pipeline_using_smc = pipeline_using_smc

    def sample_images(self, train=False):
        """Sample images using the diffusion model."""
        self.pipeline.unet.eval()

        num_particles = int(self.config.smc.num_particles)
        prompts_all = list(self.eval_prompts)

        # print(f"Eval Prompts: {prompts_all}")

        with torch.no_grad():
            for prompt in tqdm(
                prompts_all,
                desc="Sampling images",
                disable=not self.accelerator.is_local_main_process,
                position=0,
            ):
                print(f"Process Prompt: {prompt}")

                # Generate a batch of images for this single prompt by running SMC multiple times.
                # The current SMC pipeline implementation is safest in `batch_size=1` mode.
                latents_0 = torch.randn(
                    (
                        num_particles,
                        self.pipeline.unet.config.in_channels,
                        self.pipeline.unet.sample_size,
                        self.pipeline.unet.sample_size,
                    ),
                    device=self.accelerator.device,
                    dtype=self.inference_dtype,
                )

                prompt_inputs = self.pipeline.tokenizer(
                    [prompt],
                    padding="max_length",
                    truncation=True,
                    max_length=self.pipeline.tokenizer.model_max_length,
                    return_tensors="pt",
                )

                prompt_embeds = self.pipeline.text_encoder(
                    prompt_inputs.input_ids.to(self.accelerator.device),
                    attention_mask=prompt_inputs.attention_mask.to(self.accelerator.device),
                )[0].to(self.inference_dtype)

                # Convert reward function to accept only images. During SMC, rewards are computed
                # for `batch_p` candidate images corresponding to the same prompt.
                image_reward_fn = lambda images, _p=prompt: self.reward_fn(
                    images,
                    [_p] * images.shape[0],
                )

                with self.autocast():
                    images, log_w, normalized_w, latents, \
                    all_log_w, resample_indices, ess_trace, \
                    scale_factor_trace, rewards_trace, manifold_deviation_trace, log_prob_diffusion_trace \
                    = self.pipeline_using_smc(
                        self.pipeline,
                        # prompt=[prompt],
                        negative_prompt=[""],
                        prompt_embeds=prompt_embeds,
                        num_inference_steps=self.config.sample.num_steps,
                        guidance_scale=self.config.sample.guidance_scale,
                        eta=self.config.sample.eta,
                        output_type="pt",
                        latents=latents_0,
                        num_particles=num_particles,
                        batch_p=num_particles,
                        resample_strategy=self.config.smc.resample_strategy,
                        ess_threshold=self.config.smc.ess_threshold,
                        tempering=self.config.smc.tempering,
                        tempering_schedule=self.config.smc.tempering_schedule,
                        tempering_gamma=self.config.smc.tempering_gamma,
                        tempering_start=self.config.smc.tempering_start,
                        reward_fn=image_reward_fn,
                        kl_coeff=self.config.smc.kl_coeff,
                        verbose=False,
                    )

                self.info_eval_vis["eval_ess"].append(ess_trace)
                self.info_eval_vis["scale_factor_trace"].append(scale_factor_trace)
                self.info_eval_vis["rewards_trace"].append(rewards_trace)
                self.info_eval_vis["manifold_deviation_trace"].append(manifold_deviation_trace)
                self.info_eval_vis["log_prob_diffusion_trace"].append(log_prob_diffusion_trace)

                rewards = self.reward_fn(images, [prompt])

                self.info_eval_vis["eval_rewards_img"].append(rewards.clone().detach())
                self.info_eval_vis["eval_image"].append(images.clone().detach())
                self.info_eval_vis["eval_prompts"].append(prompt)

                if self.accelerator.is_local_main_process:
                    reward_value = rewards.mean().detach().cpu().item()
                    image_tensor = images[0].detach().float().cpu()
                    image_tensor = ((image_tensor.clamp(-1, 1) + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                    image_pil = Image.fromarray(image_tensor.permute(1, 2, 0).numpy())

                    log_payload = {
                        "eval/current_reward": reward_value,
                        "eval/current_prompt": prompt,
                    }

                    if wandb.run is not None:
                        log_payload["eval/current_image"] = wandb.Image(image_pil, caption=prompt)
                        wandb.log(log_payload)
                    else:
                        print(f"Prompt: {prompt} | Reward: {reward_value}")
        
        # ### Stats?
        # eval_images_stats = self.info_eval_vis["eval_image"]
        # print(f"Eval Image Length: {len(eval_images_stats)}")

    def log_evaluation(self, epoch=None, inner_epoch=None):
        rewards = torch.cat(self.info_eval_vis["eval_rewards_img"])
        prompts = self.info_eval_vis["eval_prompts"]
        # ess_trace = torch.cat(self.info_eval_vis["eval_ess"])
        # scale_factor_trace = torch.cat(self.info_eval_vis["scale_factor_trace"])
        # rewards_trace = torch.cat(self.info_eval_vis["rewards_trace"])
        # manifold_deviation_trace = torch.cat(self.info_eval_vis["manifold_deviation_trace"])
        # log_prob_diffusion_trace = torch.cat(
        #     self.info_eval_vis["log_prob_diffusion_trace"]
        # )

    # def log_evaluation(self, epoch=None, inner_epoch=None):
    #     super().log_evaluation(epoch=None, inner_epoch=None)

    #     rewards = torch.cat(self.info_eval_vis["eval_rewards_img"])
    #     prompts = self.info_eval_vis["eval_prompts"]

    #     ess_trace = torch.cat(self.info_eval_vis["eval_ess"])
    #     scale_factor_trace = torch.cat(self.info_eval_vis["scale_factor_trace"])
    #     rewards_trace = torch.cat(self.info_eval_vis["rewards_trace"])
    #     manifold_deviation_trace = torch.cat(self.info_eval_vis["manifold_deviation_trace"])
    #     log_prob_diffusion_trace = torch.cat(self.info_eval_vis["log_prob_diffusion_trace"])
        
    #     for i, ess in enumerate(ess_trace):

    #         fig, ax1 = plt.subplots()
    #         ax2 = ax1.twinx()

    #         ax1.plot(range(len(ess)), ess, 'b-')
    #         caption = f"{i:03d}_{prompts[i]} | reward: {rewards[i]}"
    #         os.makedirs(f"{self.log_dir}/{caption}", exist_ok=True)

    #         plt.savefig(f"{self.log_dir}/{caption}/ess.png")
    #         plt.clf()

    #         plt.plot(rewards_trace[i])
    #         plt.savefig(f"{self.log_dir}/{caption}/intermediate_rewards.png")
    #         plt.clf()

    #         plt.plot(manifold_deviation_trace[i])
    #         plt.savefig(f"{self.log_dir}/{caption}/manifold_deviation.png")
    #         plt.clf()

    #         plt.plot(log_prob_diffusion_trace[i])
    #         plt.savefig(f"{self.log_dir}/{caption}/log_prob_diffusion.png")
    #         plt.clf()

    #         np.save(f"{self.log_dir}/{caption}/ess.npy", ess)
    #         np.save(f"{self.log_dir}/{caption}/manifold_deviation.npy", manifold_deviation_trace[i])
    #         np.save(f"{self.log_dir}/{caption}/log_prob_diffusion.npy", log_prob_diffusion_trace[i])

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/sd.py", "Sampling configuration.")

def main(_):
    # Load the configuration
    config = FLAGS.config

    # Initialize the trainer with the configuration
    sampler = DAS(config)

    # Run sampling
    sampler.run_evaluation()

if __name__ == "__main__":
    app.run(main)
