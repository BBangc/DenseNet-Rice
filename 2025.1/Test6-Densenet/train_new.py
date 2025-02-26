# import os
# import math
# import argparse
#
# import torch
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
# import torch.optim.lr_scheduler as lr_scheduler
#
# from model import densenet121, load_state_dict
# from my_dataset import MyDataSet
# from utils import read_split_data, train_one_epoch, evaluate
#
#
# def main(args):
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#
#     print(args)
#     print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
#     tb_writer = SummaryWriter()
#     if os.path.exists("./weights") is False:
#         os.makedirs("./weights")
#
#     train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
#
#     data_transform = {
#         "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#         "val": transforms.Compose([transforms.Resize(256),
#                                    transforms.CenterCrop(224),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#
#     # 实例化训练数据集
#     train_dataset = MyDataSet(images_path=train_images_path,
#                               images_class=train_images_label,
#                               transform=data_transform["train"])
#
#     # 实例化验证数据集
#     val_dataset = MyDataSet(images_path=val_images_path,
#                             images_class=val_images_label,
#                             transform=data_transform["val"])
#
#     batch_size = args.batch_size
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#     print('Using {} dataloader workers every process'.format(nw))
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                pin_memory=True,
#                                                num_workers=nw,
#                                                collate_fn=train_dataset.collate_fn)
#
#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              pin_memory=True,
#                                              num_workers=nw,
#                                              collate_fn=val_dataset.collate_fn)
#
#     # 如果存在预训练权重则载入
#     model = densenet121(num_classes=args.num_classes).to(device)
#     if args.weights != "":
#         if os.path.exists(args.weights):
#             load_state_dict(model, args.weights)
#         else:
#             raise FileNotFoundError("not found weights file: {}".format(args.weights))
#
#     # 是否冻结权重
#     if args.freeze_layers:
#         for name, para in model.named_parameters():
#             # 除最后的全连接层外，其他权重全部冻结
#             if "classifier" not in name:
#                 para.requires_grad_(False)
#
#     pg = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
#     # Scheduler https://arxiv.org/pdf/1812.01187.pdf
#     lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
#
#     for epoch in range(args.epochs):
#         # train
#         mean_loss = train_one_epoch(model=model,
#                                     optimizer=optimizer,
#                                     data_loader=train_loader,
#                                     device=device,
#                                     epoch=epoch)
#
#         scheduler.step()
#
#         # validate
#         acc = evaluate(model=model,
#                        data_loader=val_loader,
#                        device=device)
#
#         print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
#         tags = ["loss", "accuracy", "learning_rate"]
#         tb_writer.add_scalar(tags[0], mean_loss, epoch)
#         tb_writer.add_scalar(tags[1], acc, epoch)
#         tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
#
#         torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_classes', type=int, default=5)
#     parser.add_argument('--epochs', type=int, default=30)
#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--lrf', type=float, default=0.1)
#
#     # 数据集所在根目录
#     # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
#     parser.add_argument('--data-path', type=str,
#                         default="/data/flower_photos")
#
#     # densenet121 官方权重下载地址
#     # https://download.pytorch.org/models/densenet121-a639ec97.pth
#     parser.add_argument('--weights', type=str, default='densenet121.pth',
#                         help='initial weights path')
#     parser.add_argument('--freeze-layers', type=bool, default=False)
#     parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
#
#     opt = parser.parse_args()
#
#     main(opt)


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

from model import densenet121

def plot_train(Tacclist, Tlosslist, Vacclist, Vlosslist, save_dir):
    data = {'accuracy': Tacclist, 'val_accuracy': Vacclist}
    pd.DataFrame(data).plot(figsize=(5, 4))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(save_dir + 'accuracy_curve_origin_2.21eve.png')
    plt.close()

    data = {'loss': Tlosslist, 'val_loss': Vlosslist}
    pd.DataFrame(data).plot(figsize=(5, 4))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_dir + 'loss_curve_origin_2.21eve.png')
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

# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight)
#         nn.init.constant_(m.bias, 0)

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
    # data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 获取数据根目录
    # image_path = os.path.join(data_root, "daogu_zazhi", "Data_new")  # 数据集路径
    # image_path = os.path.join(data_root, "Data")
    assert os.path.exists(train_image_path), "{} 路径不存在。".format(train_image_path)
    assert os.path.exists(val_image_path), "{} 路径不存在。".format(val_image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(train_image_path, "train"),
                                   transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(train_image_path, "test"),
                                         transform=data_transform["test"])
    # train_dataset, val_dataset = split_train_val(dataset)
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('每个进程使用 {} 个数据加载器进程'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("使用 {} 张图像进行训练，{} 张图像进行验证。".format(train_num, val_num))

    net = densenet121(num_classes=len(class_list)).to(device)
    # # 重新初始化模型权重
    # net.apply(weights_init)

    # 不加载预训练权重，直接从头开始训练
    # 如果需要冻结层，可以取消注释以下部分
    # freeze_layers = False
    # if freeze_layers:
    #     for name, param in net.named_parameters():
    #         if "classifier" not in name:
    #             param.requires_grad_(False)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()

    # 构建优化器
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './densenet121_origin_2.21eve.pth'
    train_steps = len(train_loader)
    train_acc_list, train_loss_list = [], []
    val_acc_list, val_loss_list = [], []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        correct, total = 0, 0
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch + 1, epochs, loss, 100 * correct / total)

        train_acc_list.append(100 * correct / total)
        train_loss_list.append(running_loss / train_steps)

        net.eval()
        acc = 0.0
        val_running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_running_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                total += val_labels.size(0)
                correct += (predict_y == val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch + 1, epochs, loss, 100 * correct / total)

        val_acc_list.append(100 * correct / total)
        val_loss_list.append(val_running_loss / len(validate_loader))

        if correct / total > best_acc:
            best_acc = correct / total
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    plot_train(train_acc_list, train_loss_list, val_acc_list, val_loss_list, './')
    save_lists_to_file('densenet_origin_2.21eve.pkl', train_acc_list, train_loss_list, val_acc_list, val_loss_list)

if __name__ == '__main__':
    main()
