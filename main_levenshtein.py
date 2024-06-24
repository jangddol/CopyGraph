# -*- coding: utf-8 -*-
import Levenshtein

from get_file import get_file_dict
from similarity import make_nonlinear_similarity, plot_2d_graph


def nonlinear_edge_weight(x : float):
    x = 1 - x
    return pow(x, 9)


def levenshtein_distance(s1, s2):
    return Levenshtein.distance(s1, s2)
    
    
if __name__ == '__main__':
    folder_path = 'D:\\Coding\\CopyGraph\\files\\FinalTuring'  # Replace with the path to your specific folder
    file_dict = get_file_dict(folder_path)
    authors = list(file_dict.keys())
    texts = list(file_dict.values())

    similarity_matrix = make_nonlinear_similarity(texts, levenshtein_distance, nonlinear_edge_weight)

    # 네트워크 그래프 시각화
    plot_2d_graph(similarity_matrix, authors)