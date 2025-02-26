import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import densenet121

# 做混淆矩阵
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_num = 3;

# 初始化混淆矩阵的实际类与预测类列表
y_true =['000-整精米'] * 218 + ['001-小碎米'] * 90 + ['002-普通碎米'] * 102;  # test集  按照文件夹内class顺序
# y_true = ['002-普通碎米'] * 0 + ['000-整精米'] * 218 + ['001-小碎米'] * 0
y_pred = [];  # 数据集的预测值

def Confuse_Matrix(y_true,y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred);
    # 可视化
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def predict_folder(folder_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = densenet121(num_classes=3).to(device)
    # load model weights
    model_weight_path = "D:\Data(E)\gaojiayuan\工作汇报\\2025.1\Test6-Densenet\densenet121_new_2.20.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # iterate over each image in the folder
    predictions = {}

    filenames = os.listdir(folder_path)
    filenames.sort(key=lambda x: int(x.split('.')[0]))  # 将总文件下的图片以序号大小排序
    for img_name in filenames:
        # print("img_name:{}".format(img_name))
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path):
            try:
                # load image
                img = Image.open(img_path)
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)

                # predict class
                with torch.no_grad():
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                # print and store result
                prediction = "class: {}   prob: {:.3}".format(
                    class_indict[str(predict_cla)], predict[predict_cla].numpy())
                print(f"Image: {img_name} - {prediction}")
                predictions[img_name] = {
                    "class": class_indict[str(predict_cla)],
                    "probability": predict[predict_cla].item()
                }
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # Print final predictions summary
    print("\nFinal Predictions Summary:")
    for img_name, pred in predictions.items():
        print(f"{img_name} - Class: {pred['class']}, Probability: {pred['probability']:.3f}")
        y_pred.append(pred['class']);

#     得到混淆矩阵
    Confuse_Matrix(y_true, y_pred);

if __name__ == '__main__':
    folder_path = "D:\Data\GJY\Paper1\Paper-Rice\Paper1\Test6-Densenet\Data\\test\\test"  # Replace with your folder path
    predict_folder(folder_path)
