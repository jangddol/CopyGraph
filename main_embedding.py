# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import numpy.typing as npt
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from similarity import make_nonlinear_similarity, plot_2d_graph
from get_file import get_file_dict


def get_embedded_vector_from_text(text) -> npt.ArrayLike:
    # Load the pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the text
    tokens: torch.List[int] = tokenizer.encode(text, add_special_tokens=True, truncation=True, padding="longest", return_tensors="pt")

    # Get the embeddings from the model
    with torch.no_grad():
        embeddings: npt.ArrayLike = model(tokens)[0].squeeze(0).numpy()

    # reshape embeddings to a long 1 dimensional array
    reshaped_embeddings: npt.ArrayLike = embeddings.reshape(-1)
    return reshaped_embeddings


def get_embedded_vector(texts) -> list[npt.ArrayLike]:
    return [get_embedded_vector_from_text(text) for text in texts]


def cosine_similarity(vector1: list, vector2: list) -> float:
    dot_product = sum([a*b for a, b in zip(vector1, vector2)])
    magnitude1 = sum([a**2 for a in vector1]) ** 0.5
    magnitude2 = sum([a**2 for a in vector2]) ** 0.5
    return dot_product / (magnitude1 * magnitude2)


def nonlinear_edge_weight(x : float) -> float:
    return 1 / (1 + np.exp(-12*(x-0.5)))


if __name__ == '__main__':
    folder_path = 'D:\\Coding\\CopyGraph\\files\\FinalTuring'  # Replace with the path to your specific folder
    file_dict: dict[str, str] = get_file_dict(folder_path)
    authors = list(file_dict.keys())
    texts = list(file_dict.values())

    embedded_vectors: list[npt.ArrayLike] = get_embedded_vector(texts)

    similarity_matrix: list[list[float]] = make_nonlinear_similarity(embedded_vectors, cosine_similarity, nonlinear_edge_weight)

    plot_2d_graph(similarity_matrix, authors)