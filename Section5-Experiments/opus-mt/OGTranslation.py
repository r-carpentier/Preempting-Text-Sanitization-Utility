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
from utils.tools import (
    print_timed,
    save_pickle,
    load_pickle,
    sampling_multi_news_texts,
)

load_dotenv()

# BEGIN PARAMETERS
cuda_device = "cuda:0"  # This model does not support multiple GPUs for inference
batch_size = 16  # Modify according to your VRAM constraints
# END PARAMETERS

save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "experiments/multi_news/translation/opusMT_translation/On_original_texts",
)
torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

# Load dataset
texts: list[str] = load_from_disk(
    join(
        os.environ["ROOT_SAVE_FOLDER"],
        "datasets/multi_news/concatenated_clean_1024_tokens",
    )
)["document"]


# SAMPLING OF TEXTS
texts = sampling_multi_news_texts(texts)

# Loading model
print_timed("Loading Language Model")
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-tc-big-en-fr",
    torch_dtype=torch.float16,  # pipeline does not seem to use the torch_dtype parameter of the config.json
    device=cuda_device,
    batch_size=batch_size,
)

# Warmup
_ = translator("In the beginning, the universe was created.", max_new_tokens=5)

# Prepare texts:
# - Tokenize the messages to know their lengths
# - Sort the messages by length to be processed in batches of approximately the same length (for performance)

# The model does not need a system prompt nor a prefix to perform its task.
messages = texts

messages_tokens = translator.tokenizer(messages, padding=False, truncation=True)[
    "input_ids"
]

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
    outputs = translator(
        batch_messages,
        max_length=max_new_tokens,  # Using max_length as the model throws a warning for max_new_tokens
        do_sample=True,
        truncation=True,
    )

    generated_translations = [output["translation_text"] for output in outputs]

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
