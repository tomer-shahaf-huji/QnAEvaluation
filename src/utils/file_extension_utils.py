import os

FILE_NAME_INDEX = 0
FILE_EXTENSION_INDEX = 1


def remove_file_extension(file_path: str) -> str:
    return os.path.splitext(file_path)[FILE_NAME_INDEX]


def extract_file_extension(file_path: str) -> str:
    return os.path.splitext(file_path)[FILE_EXTENSION_INDEX]
