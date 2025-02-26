import os

# 指定文件夹路径
folder_path = "D:\Data\GJY\Paper1\Paper-Rice\Paper1\Test6-Densenet\Data\\test\\002-普通碎米"

# 获取文件夹中所有文件名
file_names = os.listdir(folder_path)

# 对文件名进行排序
sorted_file_names = sorted(file_names)

# 重命名文件
for idx, old_name in enumerate(sorted_file_names):
    # 构造新的文件名
    new_name = f"{idx+309}.jpg"  # 假设文件都是jpg格式，可以根据实际情况修改后缀名
    # 拼接原文件路径和新文件路径
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)
    # 重命名文件
    os.rename(old_path, new_path)
