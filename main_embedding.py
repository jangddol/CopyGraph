# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

from similarity import make_nonlinear_similarity, plot_2d_graph
from get_file import get_file_dict


def get_embedded_vector_from_text(text):# -> ndarray:
    # Load the pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, padding="longest", return_tensors="pt")

    # Get the embeddings from the model
    with torch.no_grad():
        embeddings = model(tokens)[0].squeeze(0)

    # Calculate the average embedding vector
    avg_embedding = torch.mean(embeddings, dim=0)

    return avg_embedding.numpy()


def get_embedded_vector(texts):# -> list[ndarray]:
    return [get_embedded_vector_from_text(text) for text in texts]


def cosine_similarity(vector1, vector2):
    # vectors a 1dim ndarray
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def nonlinear_edge_weight(x : float):
    # return 1 / (1 + np.exp(-0.5*(x-0.5)))
    return pow(x, 9)


if __name__ == '__main__':
    folder_path = 'D:\\Coding\\CopyGraph\\files\\Lab3'  # Replace with the path to your specific folder
    file_dict = get_file_dict(folder_path)
    authors = list(file_dict.keys())
    texts = list(file_dict.values())

    embedded_vectors = get_embedded_vector(texts)

    similarity_matrix = make_nonlinear_similarity(embedded_vectors, cosine_similarity, nonlinear_edge_weight)

    plot_2d_graph(similarity_matrix, authors)