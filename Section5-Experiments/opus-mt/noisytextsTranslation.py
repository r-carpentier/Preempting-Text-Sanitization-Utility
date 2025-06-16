from transformers import pipeline
from datetime import datetime
import re
import numpy as np
import torch
from dotenv import load_dotenv
import os
from os.path import join
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import print_timed, save_pickle, load_pickle

load_dotenv()

# BEGIN PARAMETERS
embedding_model = "bart"  # "bart" or "llama", depending on the embedding model that was used to sanitize texts
cuda_device = "cuda:0"  # This model does not support multiple GPUs for inference
batch_size = 32  # Modify according to your VRAM constraints
# END PARAMETERS

load_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"experiments/multi_news/noisy_texts/{embedding_model}_embedding_model",
)
save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"experiments/multi_news/translation/opusMT_translation/On_noisy_texts_with_{embedding_model}_embedding_model",
)

torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

# Load language model

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


for file in sorted(os.listdir(load_folderpath)):
    regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
    if not regexp_match:
        continue
    epsilon = int(regexp_match.group(1))
    print_timed(f"Epsilon={epsilon}")
    texts: list[str] = load_pickle(load_folderpath, f"epsi{epsilon}full.pickle")
    n = len(texts)

    # Prepare texts:
    # - Tokenize the messages to know their lengths
    # - Sort the messages by length to be processed in batches of approximately the same length (for performance)

    # The model does not need a system prompt nor a prefix to perform its task.
    messages = texts

    messages_tokens = translator.tokenizer(messages, padding=False, truncation=True)[
        "input_ids"
    ]

    # Get the length of each message
    messages_lengths = np.array(
        [len(message_tokens) for message_tokens in messages_tokens]
    )
    # Argsort: Keep a trace of the original order of the messages to be restored later
    messages_indexes_sorted = messages_lengths.argsort()
    # Sort the messages by length. Also sort the array of lengths
    messages = [messages[i] for i in messages_indexes_sorted]
    messages_lengths = messages_lengths[messages_indexes_sorted]

    # Perform translation

    filenames_prefix = f"epsi{epsilon}"
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
            save_folderpath,
            f"{filenames_prefix}part_{filenumber:04d}.pickle",
            generated_translations,
            False,
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
        if re.fullmatch(f"^{filenames_prefix}part.*.pickle", file):
            generated_translations += load_pickle(save_folderpath, file)

    # Restore the original order
    generated_translations = [
        generated_translations[i] for i in messages_indexes_sorted.argsort()
    ]

    save_pickle(
        save_folderpath, f"{filenames_prefix}full.pickle", generated_translations, False
    )
