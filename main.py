import os
from docx import Document
from PyPDF2 import PdfReader
from docx.document import Document
import numpy as np

import matplotlib.pyplot as plt

def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.docx') or file.endswith('.pdf'):
                file_paths.append(os.path.join(root, file))
    return file_paths

def get_embedded_vector(file_path) -> None:
    if file_path.endswith('.docx'):
        doc: Document = Document(file_path)
        # Extract text from docx file
        text: str = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        # Extract embedded vector from text
        embedded_vector = None
    elif file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            # Extract text from pdf file
            text = ' '.join([page.extract_text() for page in pdf.pages])
            # Extract embedded vector from text
            embedded_vector = None  
    return embedded_vector

def plot_2d_graph(vectors):
    # Perform 2D approximation of the vectors
    approximated_vectors = None  # Replace with your code to perform the approximation

    # Plot the approximated vectors
    x = approximated_vectors[:, 0]
    y = approximated_vectors[:, 1]
    plt.scatter(x, y)
    plt.show()

if __name__ == '__main__':
    folder_path = 'D:\\Coding\\CopyGraph\\files'  # Replace with the path to your specific folder
    file_paths = get_file_paths(folder_path)

    embedded_vectors = []
    for file_path in file_paths:
        embedded_vector = get_embedded_vector(file_path)
        if embedded_vector is not None:
            embedded_vectors.append(embedded_vector)

    plot_2d_graph(embedded_vectors)