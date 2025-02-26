from torch.profiler import profile, ProfilerActivity, record_function
import os
import sys
import json
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time

from model import densenet121

def plot_train(Tacclist, Tlosslist, Vacclist, Vlosslist, save_dir):
    data = {'accuracy': Tacclist, 'val_accuracy': Vacclist}
    pd.DataFrame(data).plot(figsize=(5, 4))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(save_dir + 'accuracy_curve_new_gjy.png')
    plt.close()

    data = {'loss': Tlosslist, 'val_loss': Vlosslist}
    pd.DataFrame(data).plot(figsize=(5, 4))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_dir + 'loss_curve_new_gjy.png')
    plt.close()

def split_train_val(dataset, val_ratio=0.2):
    train_idx, val_idx = [], []
    for i in range(len(dataset)):
        if i % int(1/val_ratio) == 0:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)

def save_lists_to_file(filename, train_acc_list, train_loss_list, val_acc_list, val_loss_list):
    data = {
        'train_acc_list': train_acc_list,
        'train_loss_list': train_loss_list,
        'val_acc_list': val_acc_list,
        'val_loss_list': val_loss_list
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


from torchvision import datasets
from torchvision.transforms import ToTensor

class CachedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CachedImageFolder, self).__init__(root, transform)
        self.cache = {}  # 图像缓存

    def __getitem__(self, index):
        path, target = self.samples[index]
        if path not in self.cache:
            self.cache[path] = Image.open(path).convert("RGB")
        img = self.cache[path]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 获取数据根目录
    train_image_path = os.path.join(data_root, "Test6-Densenet", "Data")  # 数据集路径
    val_image_path = os.path.join(data_root, "Test6-Densenet", "Data")  # 数据集路径

    assert os.path.exists(train_image_path), "{} 路径不存在。".format(train_image_path)
    assert os.path.exists(val_image_path), "{} 路径不存在。".format(val_image_path)
    
    

    batch_size = 16

    train_dataset = CachedImageFolder(train_image_path, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)

    
    
    
    
    train_dataset = datasets.ImageFolder(root=os.path.join(train_image_path, "train"),
                                   transform=data_transform["train"])

    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('每个进程使用 {} 个数据加载器进程'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, pin_memory=True)

    net = densenet121(num_classes=len(class_list)).to(device)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()

    # 构建优化器
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)



    # 测试cpu到gpu的传输延迟
    torch.cuda.synchronize()
    start = time.time()
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        break
    torch.cuda.synchronize()
    end = time.time()
    print(f"Data transfer time: {end - start:.5f} seconds")



    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        if i > 10:
            break
    print(f"加载 10 批次时间：{time.time() - start:.2f} 秒")










    # pytorch_profilling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True
    ) as prof:
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = net(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            if step == 10:  # 测试 10 个 batch
                break

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == '__main__':
    main()
