import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('pretrained', type=str)
args = parser.parse_args()

pretrained = args.pretrained

tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float16)


dataset = load_dataset('EleutherAI/wikitext_document_level', 'wikitext-2-raw-v1')
dataset = load_dataset('piqa', 'plain_text')
dataset = load_dataset('hellaswag')
dataset = load_dataset('winogrande', 'winogrande_xl')

