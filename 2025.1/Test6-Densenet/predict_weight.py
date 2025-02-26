import os
import json
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import shutil
import cv2
import numpy as np

from model_new import densenet121  # 确保 model.py 文件中包含 densenet121 模型的定义
# from model_moreDeconv_addlayer import densenet121
# from model_Deconv import densenet121

# 定义用于计算轮廓面积的函数
def countNum(img_path):
    original_filename = os.path.basename(img_path)
    img_dir = os.path.dirname(img_path)
    font = cv2.FONT_HERSHEY_COMPLEX
    kernel = np.ones((7, 7), np.uint8)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_area_total = 0

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        area = cv2.contourArea(cnt)
        contour_area_total += area
        area_circle = 3.14 * radius * radius

        if area < 100:
            continue
        if area / area_circle <= 0.5:
            img = cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
            scale_factor = 1.5
            new_width = int(img.shape[1] * scale_factor)
            new_height = int(img.shape[0] * scale_factor)
            new_size = (new_width, new_height)
            resized_img = cv2.resize(img, new_size)
            text_size, _ = cv2.getTextSize(str(area), font, 0.4, 1)
            text_x = 10
            text_y = text_size[1] + 10
            img = cv2.putText(resized_img, str(area), (text_x, text_y), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            new_filename = f"{original_filename[:-4]}_area.jpg"
            new_save_path = os.path.join(img_dir, new_filename)
            cv2.imwrite(new_save_path, img)
        elif area / area_circle >= 0.6:
            img = cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
            scale_factor = 3
            new_width = int(img.shape[1] * scale_factor)
            new_height = int(img.shape[0] * scale_factor)
            new_size = (new_width, new_height)
            resized_img = cv2.resize(img, new_size)
            text_size, _ = cv2.getTextSize(str(area), font, 0.4, 1)
            text_x = 10
            text_y = text_size[1] + 10
            img = cv2.putText(resized_img, str(area), (text_x, text_y), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            new_filename = f"{original_filename[:-4]}_area.jpg"
            new_save_path = os.path.join(img_dir, new_filename)
            cv2.imwrite(new_save_path, img)
        else:
            img = cv2.drawContours(img, cnt, -1, (255, 0, 0), 3)
            scale_factor = 3
            new_width = int(img.shape[1] * scale_factor)
            new_height = int(img.shape[0] * scale_factor)
            new_size = (new_width, new_height)
            resized_img = cv2.resize(img, new_size)
            text_size, _ = cv2.getTextSize(str(area), font, 0.5, 1)
            text_x = 10
            text_y = text_size[1] + 10
            img = cv2.putText(resized_img, str(area), (text_x, text_y), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            new_filename = f"{original_filename[:-4]}_area.jpg"
            new_save_path = os.path.join(img_dir, new_filename)
            cv2.imwrite(new_save_path, img)

    return contour_area_total


def predict(model, img_path, classNames):
    img = Image.open(img_path)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)

    model.eval()
    with torch.no_grad():
        output = model(img)
        _, indices = torch.max(output, 1)
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        perc = percentage[int(indices)].item()
        result = classNames[indices]
        print('predicted:', result, perc)

        if result == '000-整精米':
            save_folder = 'predict_outcome/000-zheng' #将“整精米”图片存入zheng文件夹
        elif result == '001-小碎米':
            save_folder = 'predict_outcome/001-xiaosuimi'
        elif result == '002-普通碎米':
            save_folder = 'predict_outcome/002-putong'
        # elif result == '干扰背景':
        #     save_folder = 'D:/csdn_example-vgg16/vgg16/Test8_densenet/predict_result/003-干扰背景'
        else:
            save_folder = 'predict_outcome/004-other'

        os.makedirs(save_folder, exist_ok=True)
        img_filename = os.path.basename(img_path)#只返回图片名
        save_path = os.path.join(save_folder, img_filename)
        shutil.copy2(img_path, save_path)#将混合米的原图复制到save_path下


if __name__ == '__main__':
    classNames = ['000-整精米', '001-小碎米', '002-普通碎米']
    num_classes = len(classNames)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = densenet121(num_classes=num_classes)
    model_weight_path = "densenet121_new_2.20.pth"
    # model_weight_path = './densenet121_deconv.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()

    img_path = 'D:\Data(E)\gaojiayuan\工作汇报\\2025.1\\fengge_code\Data2.21\\test\\000-整精米'  #待预测文件夹
    root = img_path

    filenames = os.listdir(root) #列出总文件夹下所有子文件
    filenames.sort(key=lambda x: int(x.split('.')[0]))  #将总文件下的图片以序号大小排序

    for path in filenames:
        childpath = os.path.join(root, path)  #一个个定位到子文件夹，子文件夹名为大米总重量
        predict(model, childpath, classNames)  #预测子文件夹下的内容，每个文件夹内应只有单张图片
        print(path)
    # ----------------------------------------------------------------------------------------------------
    folders_to_check = {
         'D:\Data\GJY\Paper1\Paper-Rice\Paper1\Test6-Densenet\predict_outcome/000-zheng': 2.520972343667084e-6,  #folder1   整精米类：质量/面积
        'D:\Data\GJY\Paper1\Paper-Rice\Paper1\Test6-Densenet\predict_outcome/001-xiaosuimi': 2.520972343667084e-6,#folder2
        'D:\Data\GJY\Paper1\Paper-Rice\Paper1\Test6-Densenet\predict_outcome/002-putong': 2.520972343667084e-6,#folder3
    }

    contour_areas = {folder: 0 for folder in folders_to_check} #contour_areas = {‘folder1’:0,'folder2':0,'folder3':0}
    image_counts = {folder: 0 for folder in folders_to_check} #image_counts = {‘folder1’:0,'folder2':0,'folder3':0}

    for folder in folders_to_check:
        if os.path.exists(folder):#如果对应文件夹存在
            if os.listdir(folder):  #如果文件夹下有图片
                for filename in os.listdir(folder): #遍历每个小物体
                    img_path = os.path.join(folder, filename)
                    contour_area = countNum(img_path)
                    contour_areas[folder] += contour_area  #求出对应类别下物体的总面积
                    image_counts[folder] += 1          #求出对应类别下物体的总数
    print(f"不同种类的总面积为{contour_areas}")
    print(f"不同种类的总数为{image_counts}")


    # 分别计算每个类别下物体的总面积和总质量
    for folder, area in contour_areas.items():
        total_area = area
        if image_counts[folder] > 0:
            average_area = total_area / image_counts[folder]  #每个小物体的平均面积
        else:
            average_area = 0  #没有检测出该类别的物体

        coefficient = folders_to_check[folder]  #系数，也就是小物体的密度
        total_mass = total_area * coefficient  #总质量=面积*密度

        print(f"{folder} 下每张图片的轮廓总面积为: {total_area}, 总质量为: {total_mass}")


    # def countNum(img_path):
    #     # 你的图像处理和轮廓面积计算逻辑
    #     pass
