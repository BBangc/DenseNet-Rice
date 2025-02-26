import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

# with open('model_convnext_daogu.pkl', 'rb') as f:
#     data = pickle.load(f)

# print(data)



# 加载 .pkl 文件
with open('densenet_origin_2.21eve.pkl', 'rb') as test1:# 新
    data_dict1 = pickle.load(test1)
with open('densenet_new_2.21eve.pkl', 'rb') as test2:# 旧
    data_dict2 = pickle.load(test2)
# with open('Test3-Resnet50/resnet50_origin_grain.pkl', 'rb') as test3:
#     data_dict3 = pickle.load(test3)
# with open('Test5-Efficientnet/efficientNet_origin_daogu.pkl', 'rb') as test5:
#     data_dict5 = pickle.load(test5)
# with open('Test6-Densenet/densenet_origin_daogu.pkl', 'rb') as test6:
#     data_dict6 = pickle.load(test6)

# 准确率字典
acc_dict = {}
acc_dict['new'] = data_dict2['train_acc_list']
acc_dict['origin'] = data_dict1['train_acc_list']

# acc_dict['Resnet50'] = data_dict3['train_acc_list']
# acc_dict['Efficientnet'] = data_dict5['train_acc_list']
# acc_dict['Densenet'] = data_dict6['train_acc_list']


# # 将test内的数据放大100倍，单位变为百分比
# for i in range(len(acc_dict['Shufflenet'])):
#     acc_dict['Shufflenet'][i] = acc_dict['Shufflenet'][i]*100
    # print(acc_dict['test2'][i])

# 假设字典的结构类似于 {'label': [value1, value2, ...]}
# 将字典转换为 DataFrame
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in acc_dict.items()]))

# 可视化数据
# 假设你想绘制每个键的平均值
df.plot(kind='line')
plt.title('Average Values of Dictionary Keys')
plt.xlabel('Keys')
plt.ylabel('Average Value')
plt.show()