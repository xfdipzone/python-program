# coding=utf-8
import shutil
from google.colab import files

"""
将 Google Colab Driver 文件夹压缩为档案并下载到本地（默认为 ZIP）
"""
def archive_and_download(dir, archive_name, format='zip'):
    archive_path = shutil.make_archive(archive_name, format, dir)
    files.download(archive_path)


# 执行归档与下载
archive_and_download('data/output_audio', 'output_audio', 'zip')
