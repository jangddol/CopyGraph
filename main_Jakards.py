# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib import rc
font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
from get_file import get_file_paths, get_authors, get_file_texts


# 자카드 유사도 계산 함수
def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def nonlinear_edge_weight(x : float):
    return 1 / (1 + np.exp(-12*(x-0.5)))


def plot_2d_graph(G: nx.Graph, names: list):
    # 네트워크 그래프 시각화
    # 2D 네트워크 그래프 시각화
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=[G[u][v]['weight']*2 for u, v in G.edges()])
    nx.draw_networkx_labels(G, pos, {i: names[i] for i in range(len(names))}, font_family=font_name, font_size=8)
    plt.title("2D Document Similarity Network")
    plt.show()


def plot_3d_graph(G: nx.Graph, texts: list):
    # 3D 네트워크 그래프 시각화
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 노드 위치 설정
    pos = nx.spring_layout(G, dim=3)
    x = [pos[i][0] for i in range(len(texts))]
    y = [pos[i][1] for i in range(len(texts))]
    z = [pos[i][2] for i in range(len(texts))]

    # 노드 및 엣지 그리기
    ax.scatter(x, y, z, c='lightblue', s=100)
    for u, v in G.edges():
        ax.plot([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], color='gray', linewidth=G[u][v]['weight']*2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Document Similarity Network")
    plt.show()


if __name__ == '__main__':
    folder_path = 'D:\\Coding\\CopyGraph\\files'  # Replace with the path to your specific folder
    file_paths = get_file_paths(folder_path)
    texts = get_file_texts(file_paths)
    # file들은 보고서이며, 보고서의 제출자를 알아내는 함수를 불러와 제출자 리스트를 만듦
    authors: list = get_authors(file_paths)

    # 문서 간 자카드 유사도 계산
    similarity_matrix = [[jaccard_similarity(text1, text2) for text2 in texts] for text1 in texts]

    # 유사도 노말라이즈
    max_similarity = max([max(similarities) for similarities in similarity_matrix])
    min_similarity = min([min(similarities) for similarities in similarity_matrix])
    similarity_matrix = [[(similarity - min_similarity) / (max_similarity - min_similarity) for similarity in similarities] for similarities in similarity_matrix]
    similarity_matrix = [[nonlinear_edge_weight(similarity) for similarity in similarities] for similarities in similarity_matrix]

    # 네트워크 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(range(len(texts)))

    # 엣지 추가 및 가중치 설정
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            G.add_edge(i, j, weight=pow(similarity_matrix[i][j], 2))

    # 네트워크 그래프 시각화
    plot_2d_graph(G, authors)