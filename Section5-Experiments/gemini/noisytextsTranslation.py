from datasets import load_from_disk
import os
import re
import time
from os.path import join
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import print_timed, save_pickle, load_pickle

load_dotenv()

# BEGIN PARAMETERS
embedding_model = "bart"  # "bart" or "llama", depending on the embedding model that was used to sanitize texts
corrected_texts = False  # Compute on correct texts or not
batch_size = 16
translate_prompt = "You are a professional English-to-French translator. Translate all English text into fluent, natural-sounding French. Maintain the original meaning, tone, and style. Do not explain or comment — output only the translated French text."
# END PARAMETERS

load_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"experiments/multi_news/noisy_texts/{embedding_model}_embedding_model",
)
save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"experiments/multi_news/translation/gemini_translation/On_noisy_texts_with_{embedding_model}_embedding_model",
)

if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Perform translation
print_timed(f"Starting Translation")
for file in sorted(os.listdir(load_folderpath)):
    regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
    if not regexp_match:
        continue
    epsilon = int(regexp_match.group(1))
    texts: list[str] = load_pickle(load_folderpath, f"epsi{epsilon}full.pickle")
    n = len(texts)

    # Prepare texts:
    # - Tokenize the texts to know their lengths
    # - Sort the texts by length to be processed in batches of approximately the same length (for performance)

    texts_lengths_filename = f"epsi{epsilon}_texts_lengths.npy"
    if not os.path.exists(join(save_folderpath, texts_lengths_filename)):
        # Count tokens. This API call is free, limited to 3000 calls per minute
        assert n <= 3000
        texts_lengths = np.zeros(n, dtype=np.uint32)

        for i in range(0, n):
            while True:
                try:
                    texts_lengths[i] = client.models.count_tokens(
                        model="gemini-2.0-flash", contents=texts[i]
                    ).total_tokens
                except (ServerError, APIError):
                    time.sleep(60)
                    continue
                break
        np.save(join(save_folderpath, texts_lengths_filename), texts_lengths)
    else:
        texts_lengths = np.load(join(save_folderpath, texts_lengths_filename))

    # Argsort: Keep a trace of the original order of the messages to be restored later
    texts_indexes_sorted = texts_lengths.argsort()
    # Sort the messages by length. Also sort the array of lengths
    texts = [texts[i] for i in texts_indexes_sorted]
    texts_lengths = texts_lengths[texts_indexes_sorted]

    filenames_prefix = f"epsi{epsilon}"
    filenumber = 0
    print_milestone = 10  # The next percentage milestone to report
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)

        texts_slice = texts[i:j]

        # Calculate max_new_tokens as 1.3 times the length of the longest message in the batch.
        # For each batch, the message having the maximum length is the last message.
        max_new_tokens = int(texts_lengths[j - 1] * 1.3)

        # Continue to call Google API in case of ServerError
        while True:
            try:
                generated_translations = [
                    client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=texts_slice[k],
                        config=types.GenerateContentConfig(
                            max_output_tokens=max_new_tokens,
                            system_instruction=translate_prompt,
                        ),
                    ).text
                    for k in range(len(texts_slice))
                ]
            except (ServerError, APIError):
                time.sleep(60)
                continue
            break

        # Save summaries to disk
        filename = f"{filenames_prefix}part_{filenumber:04d}.pickle"
        save_pickle(save_folderpath, filename, generated_translations, False)
        filenumber += 1

        # Print progression
        percent_done = (j / n) * 100
        if percent_done >= print_milestone:  # If we passed a milestone
            print_timed(f"Epsilon={epsilon} ; Processed {int(percent_done)}% ({j}/{n})")
            print_milestone += 10  # Move to the next milestone

    # Merge results
    generated_translations = []
    for file in sorted(os.listdir(save_folderpath)):
        if re.fullmatch(f"^{filenames_prefix}part.*", file):
            generated_translations += load_pickle(save_folderpath, file)

    # Restore the original order
    generated_translations = [
        generated_translations[i] for i in texts_indexes_sorted.argsort()
    ]

    save_pickle(
        save_folderpath, f"{filenames_prefix}full.pickle", generated_translations, False
    )
