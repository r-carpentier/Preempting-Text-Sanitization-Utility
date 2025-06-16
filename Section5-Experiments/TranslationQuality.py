from datasets import load_from_disk
from dotenv import load_dotenv
import os
from os.path import join
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import re

sys.path.append("../")  # Add parent directory to path
from utils.tools import print_timed, save_pickle, load_pickle, sampling_multi_news_texts

load_dotenv()

# This script computes all the translation qualities presented in the paper.
# BEGIN PARAMETERS
embedding_models = ["bart", "llama"]
language_models = ["opusMT", "llama", "llama3.2-1B", "t5", "gemini"]
cuda_device = "cuda"
# END PARAMETERS

noisy_texts_root_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"], "experiments/multi_news/noisy_texts"
)
translations_root_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "experiments/multi_news/translation/",
)
save_folderpath = join(translations_root_folderpath, "similarities/")
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

# Load quality evaluation model

similarity_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


def compute_similarities(A: np.ndarray, B: np.ndarray) -> list[int]:
    assert A.shape[0] == B.shape[0], "Inputs should have equal first dimension"
    return [similarity_model.similarity(A[i], B[i]).item() for i in range(A.shape[0])]


def save_similarities(filename: str, similarities) -> None:
    save_pickle(save_folderpath, filename, similarities)


# Load multi_news and compute embeddings of OG texts
mnews = load_from_disk(
    join(
        os.environ["ROOT_SAVE_FOLDER"],
        "datasets/multi_news/concatenated_clean_1024_tokens",
    )
)

og_texts = sampling_multi_news_texts(mnews["document"])

# Compute embeddings of OG texts
og_texts_embeddings = similarity_model.encode(og_texts, show_progress_bar=False)

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
            noisy_texts_embedding = similarity_model.encode(
                noisy_texts, show_progress_bar=False
            )

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
# # OG texts VS Gen Translation
# Similarity between the original text and a translation generated from said text

for language_model in language_models:
    gentranslations_folderpath = join(
        translations_root_folderpath,
        f"{language_model}_translation/On_original_texts",
    )

    gentranslations = load_pickle(gentranslations_folderpath, "full.pickle")
    gentranslations = np.array(gentranslations)

    gentranslations_embedding = similarity_model.encode(
        gentranslations, show_progress_bar=False
    )
    similarities = compute_similarities(og_texts_embeddings, gentranslations_embedding)

    save_similarities(f"OGtextsVS{language_model}Gentranslation.pickle", similarities)


##################################################################################
# # OG text VS noisy gen translation
# Similarity between the original text and a translation generated from a sanitized version of said text.

for language_model in language_models:
    for embedding_model in embedding_models:
        noisy_generated_translations_folderpath = join(
            translations_root_folderpath,
            f"{language_model}_translation/On_noisy_texts_with_{embedding_model}_embedding_model",
        )

        similarities: dict[int, list[float]] = {}

        for file in sorted(os.listdir(noisy_generated_translations_folderpath)):
            regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
            if not regexp_match:
                continue
            epsilon = int(regexp_match.group(1))

            noisy_generated_translations = load_pickle(
                noisy_generated_translations_folderpath, f"epsi{epsilon}full.pickle"
            )

            noisy_generated_translations_embedding = similarity_model.encode(
                noisy_generated_translations, show_progress_bar=False
            )
            similarities[epsilon] = compute_similarities(
                og_texts_embeddings, noisy_generated_translations_embedding
            )

        save_similarities(
            f"OGtextVS{language_model}noisygenTranslationFrom{embedding_model}_embedding_model.pickle",
            similarities,
        )

# Additionnal code to handle corrected texts

# In the paper, only the following language models can be tested against corrected texts
correction_enabled_language_models = ["llama3.2-1B", "llama", "gemini"]
# Keep language models that are both correction_enabled and in the selected language models
for language_model in set(correction_enabled_language_models) & set(language_models):
    for embedding_model in embedding_models:
        noisy_generated_translations_folderpath = join(
            translations_root_folderpath,
            f"{language_model}_translation/On_noisy_texts_with_{embedding_model}_embedding_model_corrected",
        )

        similarities: dict[int, list[float]] = {}

        for file in sorted(os.listdir(noisy_generated_translations_folderpath)):
            regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
            if not regexp_match:
                continue
            epsilon = int(regexp_match.group(1))

            noisy_generated_translations = load_pickle(
                noisy_generated_translations_folderpath, f"epsi{epsilon}full.pickle"
            )

            noisy_generated_translations_embedding = similarity_model.encode(
                noisy_generated_translations
            )
            similarities[epsilon] = compute_similarities(
                og_texts_embeddings, noisy_generated_translations_embedding
            )

        save_similarities(
            f"OGtextVS{language_model}noisygenTranslationFrom{embedding_model}_embedding_model_corrected.pickle",
            similarities,
        )
