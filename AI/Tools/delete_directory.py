# coding=utf-8
import shutil

"""
删除指定目录及其所有子目录
"""
def delete_directory(dir):
    try:
        shutil.rmtree(dir)
        print(f"成功删除目录: {dir}")
    except FileNotFoundError:
        print(f"目录不存在: {dir}")
    except PermissionError:
        print(f"没有权限删除目录: {dir}")
    except Exception as e:
        print(f"删除失败: {e}")


# 执行删除
dir = './data/index_mr_fujino'
delete_directory(dir)
