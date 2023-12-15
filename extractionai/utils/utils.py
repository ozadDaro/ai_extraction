import openai
import json
from tqdm import tqdm



def init_openai(config):
    for key, value in config.items():
        setattr(openai, key, value)


def load_config(path):
    with open(path) as f:
        js = json.load(f)
        return js

