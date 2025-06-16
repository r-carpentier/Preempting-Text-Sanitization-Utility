from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv
import os
import re
from os.path import join
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))  # Add parent directory to path
from utils.tools import save_pickle, load_pickle, sampling_multi_news_texts

load_dotenv()

# This script computes all the summarization similarities presented in the paper.
# BEGIN PARAMETERS
embedding_models = ["bart", "llama"]
language_models = ["bart", "llama", "llama3.2-1B", "t5", "gemini"]
cuda_device = "cuda"
# END PARAMETERS

torch.set_default_device(cuda_device)
noisy_texts_root_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"], "experiments/multi_news/noisy_texts"
)
summaries_root_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "experiments/multi_news/summarization/",
)
save_folderpath = join(summaries_root_folderpath, "similarities/")
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)


def compute_similarities(
    embeddingsA: np.ndarray, embeddingsB: np.ndarray
) -> list[float]:
    sizeA = embeddingsA.shape[0]
    sizeB = embeddingsB.shape[0]
    assert sizeA == sizeB, "Inputs should have equal first dimension"
    similarities = [
        util.cos_sim(embeddingsA[i], embeddingsB[i]).item() for i in range(sizeA)
    ]
    return similarities


def save_similarities(filename: str, similarities) -> None:
    save_pickle(save_folderpath, filename, similarities)


# Load embedding model
similarity_model = SentenceTransformer(
    "all-mpnet-base-v2",
    device=cuda_device,
    revision="9a3225965996d404b775526de6dbfe85d3368642",
)

# Load original texts
mnews = load_from_disk(
    join(
        os.environ["ROOT_SAVE_FOLDER"],
        "datasets/multi_news/concatenated_clean_1024_tokens",
    )
)
# Compute embeddings of a sample of original texts
og_texts = sampling_multi_news_texts(mnews["document"])
og_texts_embeddings = similarity_model.encode(og_texts)

##################################################################################
# # OG texts VS noisy texts
# Similarity between the original texts and a sanitized version of said texts

for embedding_model in embedding_models:
    for corrected_text in [False, True]:
        if corrected_text:
            foldername = f"{embedding_model}_embedding_model_corrected"
        else:
            foldername = f"{embedding_model}_embedding_model"
        noisy_texts_folderpath = join(noisy_texts_root_folderpath, foldername)

        similarities: dict[int, list[float]] = {}

        # Load file for each epsilon value
        for file in sorted(os.listdir(noisy_texts_folderpath)):
            regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
            if not regexp_match:
                continue
            epsilon = int(regexp_match.group(1))

            # Encode the noisy texts
            noisy_texts = load_pickle(
                noisy_texts_folderpath, f"epsi{epsilon}full.pickle"
            )
            noisy_texts_embedding = similarity_model.encode(noisy_texts)

            # Compute similarities with original texts
            similarities[epsilon] = compute_similarities(
                og_texts_embeddings, noisy_texts_embedding
            )

        if corrected_text:
            save_similarities(
                f"OGtextsVSnoisytextsFrom{embedding_model}_embedding_model_corrected.pickle",
                similarities,
            )
        else:
            save_similarities(
                f"OGtextsVSnoisytextsFrom{embedding_model}_embedding_model.pickle",
                similarities,
            )

##################################################################################
# # OG texts VS gen summaries
# Similarity between the original text and a summary generated from said text

for language_model in language_models:
    gensummaries_folderpath = join(
        summaries_root_folderpath,
        f"{language_model}_generated_summaries/On_original_texts",
    )

    gensummaries = load_pickle(gensummaries_folderpath, "full.pickle")
    gensummaries = np.array(gensummaries)

    gensummaries_embedding = similarity_model.encode(gensummaries)
    similarities = compute_similarities(og_texts_embeddings, gensummaries_embedding)

    save_similarities(f"OGtextsVS{language_model}Gensummary.pickle", similarities)

##################################################################################
# # OG text VS noisy gen summary
# Similarity between the original text and a summary generated from a sanitized version of said text.

for language_model in language_models:
    for embedding_model in embedding_models:
        noisy_generated_summaries_folderpath = join(
            summaries_root_folderpath,
            f"{language_model}_generated_summaries/On_noisy_texts_with_{embedding_model}_embedding_model",
        )

        similarities: dict[int, list[float]] = {}

        for file in sorted(os.listdir(noisy_generated_summaries_folderpath)):
            regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
            if not regexp_match:
                continue
            epsilon = int(regexp_match.group(1))

            noisy_generated_summaries = load_pickle(
                noisy_generated_summaries_folderpath, f"epsi{epsilon}full.pickle"
            )

            noisy_generated_summaries_embedding = similarity_model.encode(
                noisy_generated_summaries
            )
            similarities[epsilon] = compute_similarities(
                og_texts_embeddings, noisy_generated_summaries_embedding
            )

        save_similarities(
            f"OGtextVS{language_model}noisygenSummaryFrom{embedding_model}_embedding_model.pickle",
            similarities,
        )

# Additionnal code to handle corrected texts

# In the paper, only the following language models can be tested against corrected texts
correction_enabled_language_models = ["llama3.2-1B", "llama", "gemini"]
# Keep language models that are both correction_enabled and in the selected language models
for language_model in set(correction_enabled_language_models) & set(language_models):
    for embedding_model in embedding_models:
        noisy_generated_summaries_folderpath = join(
            summaries_root_folderpath,
            f"{language_model}_generated_summaries/On_noisy_texts_with_{embedding_model}_embedding_model_corrected",
        )

        similarities: dict[int, list[float]] = {}

        for file in sorted(os.listdir(noisy_generated_summaries_folderpath)):
            regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
            if not regexp_match:
                continue
            epsilon = int(regexp_match.group(1))

            noisy_generated_summaries = load_pickle(
                noisy_generated_summaries_folderpath, f"epsi{epsilon}full.pickle"
            )

            noisy_generated_summaries_embedding = similarity_model.encode(
                noisy_generated_summaries
            )
            similarities[epsilon] = compute_similarities(
                og_texts_embeddings, noisy_generated_summaries_embedding
            )

        save_similarities(
            f"OGtextVS{language_model}noisygenSummaryFrom{embedding_model}_embedding_model_corrected.pickle",
            similarities,
        )
