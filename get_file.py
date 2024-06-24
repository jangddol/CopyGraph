import os 
from docx import Document
from PyPDF2 import PdfReader

def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.docx') or file.endswith('.pdf'):
                file_paths.append(os.path.join(root, file))
    return file_paths


def get_author(file_path):
    # file_path의 파일명에서 첫 underscore 이후 나오는 열개의 숫자를 가지고 학번을 인식,
    # 학번이 중간에 포함된 파일을 찾아 그 중 text 파일이 있으면 첫줄만 읽어옮.
        # text 파일의 name는 기존의 file_name에서 3번째 underscore 이후의 문자열을 지우고, 3번째 underscore도 지우고 뒤에 .txt를 붙이면 됨.
    # 첫줄의 형식은 다음과 같음 "Name: {Name} . ({student ID:10 digits})"
    # 이를 통해 제출자의 이름을 추출
    # 추출한 이름을 authors 리스트에 추가
    file_name = os.path.basename(file_path)
    student_id = file_name.split('_')[1][:10]
    text_file_name = file_name.split('_')[0] + '_' + file_name.split('_')[1] + '_' + file_name.split('_')[2] + '_' + file_name.split('_')[3] + '.txt'
    text_file_path = os.path.join(os.path.dirname(file_path), text_file_name)
    if os.path.exists(text_file_path):
        try:
            # UTF-8 인코딩 시도
            with open(text_file_path, 'r', encoding='utf-8') as file:
                author: str = file.readline().split(':')[1].strip().split('.')[0]
        except UnicodeDecodeError:
            try:
                # CP949 인코딩 시도
                with open(text_file_path, 'r', encoding='cp949') as file:
                    author = file.readline().split(':')[1].strip().split('.')[0]
            except UnicodeDecodeError:
                # 둘 다 실패하면 예외 발생
                raise Exception(f"Failed to decode file: {text_file_path}")
    return author + " " + student_id


def get_file_text(file_path) -> str:
    if file_path.endswith('.docx'):
        doc = Document(file_path)
        # Extract text from docx file
        text: str = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    elif file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            # Extract text from pdf file
            text = ' '.join([page.extract_text() for page in pdf.pages])
    else:
        raise Exception(f"Unsupported file type: {file_path}")
    return text


# def get_file_texts(file_paths):# -> list:
#     texts = []
#     for file_path in file_paths:
#         texts(get_file_text(file_path))
#     return texts


# def get_authors(file_paths):# -> list:
#     authors = []
#     for file_path in file_paths:
#         author = get_author(file_path)
#         authors.append(author)
#     return authors


def get_file_dict(folder_path):# -> dict:
    file_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.docx') or file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                author = get_author(file_path)
                text = get_file_text(file_path)
                # author can have multiple files
                # if then, join the texts with a space
                if author in file_dict:
                    file_dict[author] += text
                else:
                    file_dict[author] = text
    return file_dict