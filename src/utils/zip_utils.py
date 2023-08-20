from zipfile import ZipFile

from src.utils.file_flags import FileFlags


def unzip_files(zip_file_path: str, unzipping_directory_path: str) -> None:
    with ZipFile(zip_file_path, FileFlags.READ) as zip_file:
        zip_file.extractall(unzipping_directory_path)
