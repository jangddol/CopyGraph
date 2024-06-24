import networkx as nx
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


def make_nonlinear_similarity(texts: list, similarity_func: callable, nonlinear_weight_func: callable):
    # 문서 간 자카드 유사도 계산
    similarity_matrix = [[similarity_func(text1, text2) for text2 in texts] for text1 in texts]

    # 유사도 노말라이즈
    max_similarity = max([max(similarities) for similarities in similarity_matrix])
    min_similarity = min([min(similarities) for similarities in similarity_matrix])
    similarity_matrix = [[(similarity - min_similarity) / (max_similarity - min_similarity) for similarity in similarities] for similarities in similarity_matrix]
    similarity_matrix = [[nonlinear_weight_func(similarity) for similarity in similarities] for similarities in similarity_matrix]
    return similarity_matrix


def plot_2d_graph(similarity_matrix: list[list[float]], authors: list):
    # 네트워크 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(range(len(authors)))

    # 엣지 추가 및 가중치 설정
    for i in range(len(authors)):
        for j in range(i+1, len(authors)):
            G.add_edge(i, j, weight=similarity_matrix[i][j])
    
    # 네트워크 그래프 시각화
    # 2D 네트워크 그래프 시각화
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=[G[u][v]['weight']*4 for u, v in G.edges()])
    nx.draw_networkx_labels(G, pos, {i: authors[i] for i in range(len(authors))}, font_family=font_name, font_size=8)
    plt.title("2D Document Similarity Network")
    plt.show()