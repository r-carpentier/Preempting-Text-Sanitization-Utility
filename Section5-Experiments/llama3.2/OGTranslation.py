from datasets import load_from_disk
import os
from os.path import join
from datetime import datetime
import torch
from transformers import pipeline
import numpy as np
import re
from dotenv import load_dotenv
from os.path import join
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import print_timed, save_pickle, load_pickle, sampling_multi_news_texts

load_dotenv()

# BEGIN PARAMETERS
cuda_device = "cuda"
batch_size = 16  # Modify according to your VRAM constraints
translate_prompt = "You are a professional English-to-French translator. Translate all English text into fluent, natural-sounding French. Maintain the original meaning, tone, and style. Do not explain or comment — output only the translated French text."
# END PARAMETERS

save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "experiments/multi_news/translation/llama3.2-1B_translation/On_original_texts",
)
torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

# Load dataset

texts: list[str] = (
    load_from_disk(
        join(
            os.environ["ROOT_SAVE_FOLDER"],
            "datasets/multi_news/concatenated_clean_1024_tokens",
        )
    )
)["document"]

# SAMPLING OF TEXTS
texts = sampling_multi_news_texts(texts)

# Load language model

print_timed("Loading Language Model")
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    batch_size=batch_size,
    token=os.environ["HUGGFACE_ACCESS_TOKEN"],
)

pipe.tokenizer.padding_side = "left"
pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

# Prepare texts:
# - Associate them with the translation system prompt
# - Tokenize the messages to know their lengths
# - Sort the messages by length to be processed in batches of approximately the same length (for performance)

messages = [
    [
        {
            "role": "system",
            "content": translate_prompt,
        },
        {"role": "user", "content": text},
    ]
    for text in texts
]

# Tokenize the messages
messages_tokens: list[list[str]] = pipe.tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, padding=False
)
# Get the length of each message
messages_lengths = np.array([len(message_tokens) for message_tokens in messages_tokens])
# Argsort: Keep a trace of the original order of the messages to be restored later
messages_indexes_sorted = messages_lengths.argsort()
# Sort the messages by length. Also sort the array of lengths
messages = [messages[i] for i in messages_indexes_sorted]
messages_lengths = messages_lengths[messages_indexes_sorted]

# Perform translation

n = len(texts)
filenumber = 0
print_milestone = 10  # The next percentage milestone to report
print_timed(f"Starting Translation")
for i in range(0, n, batch_size):
    j = min(i + batch_size, n)
    batch_messages = messages[i:j]

    # Calculate max_new_tokens as 1.3 times the length of the longest message in the batch.
    # For each batch, the message having the maximum length is the last message.
    max_new_tokens = int(messages_lengths[j - 1] * 1.3)

    # Perform translation
    outputs = pipe(
        batch_messages,
        max_new_tokens=max_new_tokens,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    generated_translations = [
        outputs[i][0]["generated_text"][-1]["content"] for i in range(len(outputs))
    ]

    # Save summaries to disk
    save_pickle(
        save_folderpath, f"part_{filenumber:04d}.pickle", generated_translations, False
    )
    filenumber += 1

    # Print progression
    percent_done = (j / n) * 100
    if percent_done >= print_milestone:  # If we passed a milestone
        print_timed(f"Processed {int(percent_done)}% ({j}/{n})")
        print_milestone += 10  # Move to the next milestone

# Merge results
generated_translations = []
for file in sorted(os.listdir(save_folderpath)):
    generated_translations += load_pickle(save_folderpath, file)

# Restore the original order
generated_translations = [
    generated_translations[i] for i in messages_indexes_sorted.argsort()
]

save_pickle(save_folderpath, "full.pickle", generated_translations, False)
