import os
import shutil
from pathlib import Path

from src.utils.file_flags import FileFlags
from src.utils.file_extension_utils import remove_file_extension


def remove_file(file_path: str) -> None:
    os.remove(file_path)


def remove_directory(directory_path: str) -> None:
    shutil.rmtree(directory_path)


def save_bytes_as_file(bytes_to_save: bytes, file_path: str) -> None:
    with open(file_path, FileFlags.WRITE_BINARY) as file:
        file.write(bytes_to_save)


def create_dir_with_filename(file_path: str) -> str:
    file_name = remove_file_extension(file_path)
    Path(file_name).mkdir(parents=True, exist_ok=True)
    return file_name
