import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from get_file import get_file_paths, get_authors, get_file_texts


def get_embedded_vector_from_text(text):# -> ndarray:
    # Load the pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, padding="longest", return_tensors="pt")

    # Get the embeddings from the model
    with torch.no_grad():
        embeddings = model(tokens)[0].squeeze(0).numpy()

    # reshape embeddings to a long 1 dimensional array
    reshaped_embeddings = embeddings.reshape(-1)
    return reshaped_embeddings


def get_embedded_vector(texts):# -> ndarray:
    return [get_embedded_vector_from_text(text) for text in texts]


def plot_2d_graph(vectors, authors):
    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Extract x and y coordinates from reduced vectors
    x = reduced_vectors[:, 0]
    y = reduced_vectors[:, 1]

    # 산점도 그리기
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y)

    # 각 점 위에 이름 표시
    for i, txt in enumerate(authors):
        ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.show()


if __name__ == '__main__':
    folder_path = 'D:\\Coding\\CopyGraph\\files'  # Replace with the path to your specific folder
    file_paths = get_file_paths(folder_path)
    texts = get_file_texts(file_paths)
    # file들은 보고서이며, 보고서의 제출자를 알아내는 함수를 불러와 제출자 리스트를 만듦
    authors: list = get_authors(file_paths)

    embedded_vectors = []
    for file_path in file_paths:
        embedded_vector = get_embedded_vector(texts)
        if embedded_vector is not None:
            embedded_vectors.append(embedded_vector)

    plot_2d_graph(embedded_vectors, authors)