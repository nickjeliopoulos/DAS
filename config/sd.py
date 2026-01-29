import ml_collections
### Deprecated, use importlib instaed
# import imp
# import importlib
import os
from config.general import general

def smc():
    config = general()
    config.project_name = "DAS_SD"
    config.smc = ml_collections.ConfigDict()

    config.smc.num_particles = 8
    config.smc.resample_strategy = "ssp"
    config.smc.ess_threshold = 0.5
    
    config.smc.tempering = "schedule" # either adaptive, FreeDoM, schedule or None
    config.smc.tempering_schedule = "exp" # either float(exponent of polynomial), "exp", or "adaptive"
    config.smc.tempering_gamma = 0.008
    config.smc.tempering_start = 0

    config.smc.verbose = False

    config.sample.num_steps = 50
    config.sample.eta = 1.0

    config.sample.batch_size = 1
    config.max_vis_images = 1

    return config

def aesthetic():
    config = smc()
    config.reward_fn = "aesthetic"
    config.prompt_fn = "eval_simple_animals"

    config.smc.kl_coeff = 0.005

    return config

def clip():
    print("CLIP Score")
    config = smc()
    config.reward_fn = "clip"
    ### ORIGINAL
    # config.prompt_fn = "eval_hps_v2_all"
    ### REPLACED with OPI
    config.prompt_fn = "open_image_prefs_60"
    
    config.smc.kl_coeff = 0.01

    return config

def multi():
    print("Aesthetic + CLIP Score")
    config = smc()
    config.reward_fn = "multi"
    config.prompt_fn = "eval_hps_v2_all"

    config.aes_weight = 1.0
    
    config.smc.kl_coeff = 0.005

    return config

def pick():
    print("PickScore")
    config = smc()
    config.reward_fn = "pick"
    config.prompt_fn = "eval_hps_v2_all"
    
    config.smc.kl_coeff = 0.0001

    return config

def imagereward():
    print("Using ImageReward")
    config = smc()
    config.reward_fn = "imagereward"
    ### ORIGINAL
    # config.prompt_fn = "eval_hps_v2_all"
    ### REPLACED with OPI
    config.prompt_fn = "open_image_prefs_60"
    
    config.smc.kl_coeff = 0.005

    return config

def get_config(name):
    return globals()[name]()
