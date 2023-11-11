def a():
    with open('E:/历史代码测试保存/报告/软件工程/CCF BDCI2021 恶意软件分类/train/000c4edbbc5a107bcf1e06b1e0f6d9a3.asm', 'rb') as file:
        byte_stream = file.read()
    print(byte_stream)
import numpy as np
import os
import pandas as pd
from PIL import Image
#参照类别表格，将文件移动到对应的文件夹中，文件夹表示类别
def make_mdkir():
    # 读取CSV文件
    df = pd.read_csv('F:/kaggledata/trainLabels.csv')

    # 获取文件夹路径
    folder_path = 'F:/kaggledata/0/train'

    # 遍历CSV文件中的每一行
    for index, row in df.iterrows():
        # 获取文件名和类别名
        filename = row['Id']
        category = str(row['Class'])
        filename2 = filename + '.bytes'
        filename = filename + '.asm'

        # 获取文件的完整路径
        file_path = os.path.join(folder_path, filename)
        file_path2 = os.path.join(folder_path, filename2)
        # 创建新的子文件夹（如果不存在）
        category_folder = os.path.join(folder_path, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

            # 将文件移动到新的子文件夹中
        if os.path.exists(file_path):
            print(file_path)
            os.rename(file_path, os.path.join(category_folder, filename))
            os.rename(file_path2, os.path.join(category_folder, filename2))


# 将.bytes中的文本16进制读出来
def read_bytes(file_path):
    res_bytes = []
    with open(file_path, mode='r') as fp:
        for lines in fp.readlines():    # 循环读出每一行
            str_bytes = lines.split(" ")  # 根据空格划分为数组
            for str_byte in str_bytes:
                if str_byte[0] == '?':  # 舍弃无用字符
                    continue
                byte = int(str_byte, 16)    # 将字符按照16进制转成数字
                if byte <= 0xFF:            # 每行前面的地址都要舍弃
                    res_bytes.append(byte)
    return res_bytes
byte_array = read_bytes('F:/kaggledata/0/train/1/0AnoOZDNbPXIr2MRBSCJ.bytes')
print(byte_array)
byte_len = len(byte_array)
file_size = int(byte_len / 1024)
print(byte_len,file_size)
image_width = 0
file_size_range = [0,10,30,60,100,200,500,1000]
image_width_range = [32,64,128,256,384,512,768,1024]
for size in file_size_range:
    if file_size > size:
        image_width = image_width_range[file_size_range.index(size)]
print(image_width)
image_height = int(byte_len / image_width)
image_bytes = np.zeros([image_height, image_width])
k = 0

for i in range(image_height):
    for j in range(image_width):
        image_bytes[i][j] = byte_array[k]
        k += 1
print(image_bytes)
# 转换成相应的二维数组
img = image_bytes
img = np.uint8(img)
# 生成图片
img = Image.fromarray(img)
img.save('F:/kaggledata/0/1.png')