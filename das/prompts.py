from importlib import resources
import os
import functools
import random
import inflect
import json

IE = inflect.engine()
ASSETS_PATH = resources.files("assets")

@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def load_json_prompts(prompt_path, max_prompts=None):
	if prompt_path.endswith(".json"):
		with open(prompt_path, "r") as f:
			data = json.load(f)
	else:
		assert prompt_path.endswith(".jsonl")
		with open(prompt_path, "r") as f:
			data = [json.loads(line) for line in f]
	assert isinstance(data, list)
	prompt_key = "prompt"
	if prompt_key not in data[0]:
		assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"

		for item in data:
			item["prompt"] = item["text"]
	if max_prompts is not None:
		data = data[:max_prompts]
	return data

def from_file(path, low=None, high=None, all=False):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {} 
    # return prompts, {}

def from_file_in_order(path, idx, low=None, high=None, all=False):
    prompts = _load_lines(path)[low:high]
    return prompts[idx % len(prompts)], {} 

def hps_v2_all(idx):
    return from_file("hps_v2_all.txt")

def simple_animals(idx):
    return from_file("simple_animals.txt")

def eval_simple_animals(idx):
    return from_file("eval_simple_animals.txt")

_opi_60_json = None
def open_image_prefs_60(idx):
    global _opi_60_json
    if _opi_60_json is None:
        _opi_60_json = load_json_prompts("assets/open_img_pref_sampled_60.jsonl")
    return _opi_60_json[idx]["prompt"]

def eval_hps_v2_all(idx):
    return from_file("hps_v2_all_eval.txt")