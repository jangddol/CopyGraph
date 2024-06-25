# -*- coding: utf-8 -*-
import numpy as np
from similarity import make_nonlinear_similarity, plot_2d_graph
from get_file import get_file_dict


# 자카드 유사도 계산 함수
def jaccard_similarity(text1, text2) -> float:
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection: int = len(set1.intersection(set2))
    union: int = len(set1.union(set2))
    return intersection / union


def nonlinear_edge_weight(x : float) -> float:
    return 1 / (1 + np.exp(-8*(x-0.5)))


if __name__ == '__main__':
    folder_path = 'D:\\Coding\\CopyGraph\\files\\FinalTuring'  # Replace with the path to your specific folder
    file_dict: dict[str, str] = get_file_dict(folder_path)
    authors = list(file_dict.keys())
    texts = list(file_dict.values())

    similarity_matrix: list[list[float]] = make_nonlinear_similarity(texts, jaccard_similarity, nonlinear_edge_weight)

    # 네트워크 그래프 시각화
    plot_2d_graph(similarity_matrix, authors)