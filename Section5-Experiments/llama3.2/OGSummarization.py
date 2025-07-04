from datasets import load_from_disk
from transformers import pipeline
from datetime import datetime
import numpy as np
import re
import torch
from dotenv import load_dotenv
import os
from os.path import join
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import print_timed, save_pickle, load_pickle, sampling_multi_news_texts

load_dotenv()

# BEGIN PARAMETERS
cuda_device = "cuda"
batch_size = 64  # Modify according to your VRAM constraints
summarize_prompt = "You are a highly skilled text summarizer. Your task is to generate concise and accurate summaries of input text. Provide only the summary itself, without any remarks, explanations, or formatting cues."
# END PARAMETERS

save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "experiments/multi_news/summarization/llama3.2-1B_generated_summaries/On_original_texts",
)
torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

texts = (
    load_from_disk(
        join(
            os.environ["ROOT_SAVE_FOLDER"],
            "datasets/multi_news/concatenated_clean_1024_tokens",
        )
    )
)["document"]

# Select the 1500 randomly sampled texts
texts = sampling_multi_news_texts(texts)

print_timed("Loading Model")
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    batch_size=batch_size,
    token=os.environ["HUGGFACE_ACCESS_TOKEN"],
    revision="9213176726f574b556790deb65791e0c5aa438b6",
)

pipe.tokenizer.padding_side = "left"
pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

messages = [
    [
        {
            "role": "system",
            "content": summarize_prompt,
        },
        {"role": "user", "content": text},
    ]
    for text in texts
]

n = len(texts)
filenumber = 0
print_milestone = 10  # The next percentage milestone to report
for i in range(0, n, batch_size):
    j = min(i + batch_size, n)
    batch_messages = messages[i:j]

    outputs = pipe(
        batch_messages,
        max_new_tokens=142,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    generated_summaries = [
        outputs[i][0]["generated_text"][-1]["content"] for i in range(len(outputs))
    ]

    # Save summaries to disk
    save_pickle(
        save_folderpath, f"filepart_{filenumber:04d}.pickle", generated_summaries, False
    )
    filenumber += 1

    # Print progression
    percent_done = (j / n) * 100
    if percent_done >= print_milestone:  # If we passed a milestone
        print_timed(f"Processed {int(percent_done)}% ({j}/{n})")
        print_milestone += 10  # Move to the next milestone

generated_summaries = []
for file in sorted(os.listdir(save_folderpath)):
    if re.fullmatch(f"^filepart_.*.pickle", file):
        generated_summaries += load_pickle(save_folderpath, file)
        os.remove(join(save_folderpath, file))

save_pickle(save_folderpath, "full.pickle", generated_summaries, False)
