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
from google.genai.errors import ServerError
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import print_timed, save_pickle, load_pickle, sampling_multi_news_texts

load_dotenv()

# BEGIN PARAMETERS
batch_size = 16
translate_prompt = "You are a professional English-to-French translator. Translate all English text into fluent, natural-sounding French. Maintain the original meaning, tone, and style. Do not explain or comment — output only the translated French text."
# END PARAMETERS
save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "experiments/multi_news/translation/gemini_translation/On_original_texts",
)
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

# Prepare texts:
# - Tokenize the texts to know their lengths
# - Sort the texts by length to be processed in batches of approximately the same length (for performance)
n = len(texts)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print_timed(f"Counting Tokens")
if not os.path.exists(join(save_folderpath, "texts_lengths.npy")):
    # Count tokens. This API call is free, limited to 3000 calls per minute
    assert n <= 3000
    texts_lengths = np.array(
        [
            client.models.count_tokens(
                model="gemini-2.0-flash", contents=text
            ).total_tokens
            for text in texts
        ]
    )
    np.save(join(save_folderpath, "texts_lengths.npy"), texts_lengths)
else:
    texts_lengths = np.load(join(save_folderpath, "texts_lengths.npy"))

# Argsort: Keep a trace of the original order of the messages to be restored later
texts_indexes_sorted = texts_lengths.argsort()
# Sort the messages by length. Also sort the array of lengths
texts = [texts[i] for i in texts_indexes_sorted]
texts_lengths = texts_lengths[texts_indexes_sorted]

# Perform translation
filenumber = 0
print_milestone = 10  # The next percentage milestone to report
print_timed(f"Starting Translation")
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
        except ServerError:
            print_timed("Server Error handled")
            time.sleep(60)
            continue
        break

    # Save summaries to disk
    save_pickle(
        save_folderpath,
        f"translation_part_{filenumber:04d}.pickle",
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
    if re.fullmatch(f"^translation_part_.*.pickle", file):
        generated_translations += load_pickle(save_folderpath, file)

# Restore the original order
generated_translations = [
    generated_translations[i] for i in texts_indexes_sorted.argsort()
]

save_pickle(save_folderpath, "full.pickle", generated_translations, False)
