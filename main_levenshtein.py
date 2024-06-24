# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein
import matplotlib.font_manager as fm
from matplotlib import rc
font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
from get_file import get_file_paths, get_authors, get_file_texts


def nonlinear_edge_weight(x : float):
    x = 1 - x
    return pow(x, 9)


def plot_2d_graph(G: nx.Graph, names: list):
    # 네트워크 그래프 시각화
    # 2D 네트워크 그래프 시각화
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=[G[u][v]['weight']*4 for u, v in G.edges()])
    nx.draw_networkx_labels(G, pos, {i: names[i] for i in range(len(names))}, font_family=font_name, font_size=8)
    plt.title("2D Document Similarity Network")
    plt.show()


# def levenshtein_distance(a: str, b: str):
#     if a == b:
#         return 0
#     a_len = len(a)
#     b_len = len(b)
#     if a == "":
#         return b_len
#     if b == "":
#         return a_len
    
#     matrix = [[] for i in range(a_len+1)]
#     for i in range(a_len+1):
#         matrix[i] = [0 for j in range(b_len+1)]
    
#     for i in range(a_len+1):
#         matrix[i][0] = i
#     for j in range(b_len+1):
#         matrix[0][j] = j
    
#     for i in range(1, a_len+1):
#         ac = a[i-1]
#         for j in range(1, b_len+1):
#             bc = b[j-1]
#             cost = 0 if (ac == bc) else 1
#             matrix[i][j] = min([matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+cost])
    
#     return matrix[a_len][b_len]


def levenshtein_distance(s1, s2):
    return Levenshtein.distance(s1, s2)
    
if __name__ == '__main__':
    folder_path = 'D:\\Coding\\CopyGraph\\files'  # Replace with the path to your specific folder
    file_paths = get_file_paths(folder_path)
    texts = get_file_texts(file_paths)
    # file들은 보고서이며, 보고서의 제출자를 알아내는 함수를 불러와 제출자 리스트를 만듦
    authors: list = get_authors(file_paths)

    # 문서 간 자카드 유사도 계산
    similarity_matrix = [[levenshtein_distance(text1, text2) for text2 in texts] for text1 in texts]

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