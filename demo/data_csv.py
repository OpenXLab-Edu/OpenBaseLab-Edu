import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('dataset/1.xlsx')
file = np.array(data)

am_1, am_2, am_3, am_4 = 0, 0, 0, 0
em_1, em_2, em_3, em_4 = 0, 0, 0, 0
ja_1, ja_2, ja_3, ja_4 = 0, 0, 0, 0
ap_1, ap_2, ap_3, ap_4 = 0, 0, 0, 0

for item in file:
    it_area = item[21]
    if it_area == 'Americas':
        it_quarter = item[7]
        if it_quarter == 1:
            am_1 += item[4]
        elif it_quarter == 2:
            am_2 += item[4]
        elif it_quarter == 3:
            am_3 += item[4]
        elif it_quarter == 4:
            am_4 += item[4]
    elif it_area == 'EMEA':
        it_quarter = item[7]
        if it_quarter == 1:
            em_1 += item[4]
        elif it_quarter == 2:
            em_2 += item[4]
        elif it_quarter == 3:
            em_3 += item[4]
        elif it_quarter == 4:
            em_4 += item[4]
    elif it_area == 'APAC':
        it_quarter = item[7]
        if it_quarter == 1:
            ap_1 += item[4]
        elif it_quarter == 2:
            ap_2 += item[4]
        elif it_quarter == 3:
            ap_3 += item[4]
        elif it_quarter == 4:
            ap_4 += item[4]
        # data_ap.append(item)
    elif it_area == 'Japan':
        it_quarter = item[7]
        if it_quarter == 1:
            ja_1 += item[4]
        elif it_quarter == 2:
            ja_2 += item[4]
        elif it_quarter == 3:
            ja_3 += item[4]
        elif it_quarter == 4:
            ja_4 += item[4]
        # data_ja.append(item)

print([am_1, ja_1, ap_1, em_1])
print([am_2, ja_2, ap_2, em_2])
print([am_3, ja_3, ap_3, em_3])
print([am_4, ja_4, ap_4, em_4])
# color_list = ['y', 'b', 'r', 'g']

# labels = ['Americas', 'Japan', 'APAC', 'EMEA']

# # yellow = [am_1, ja_1, ap_1, em_1]
# # blue = [am_2, ja_2, ap_2, em_2]
# # green = [am_3, ja_3, ap_3, em_3]
# # red = [am_4, ja_4, ap_4, em_4]
# # err = [1, 1, 1, 1]

# # width = 0.5

# fig, ax = plt.subplots()

# # ax.bar(labels, blue, width, yerr=err, bottom=[yellow, blue, green, red], label='blue')
# # ax.bar(labels, yellow, width, yerr=err, bottom=[yellow, blue, green, red], label='yellow')
# # ax.bar(labels, red, width, yerr=err, bottom=[yellow, blue, green, red], label='red')
# # ax.bar(labels, green, width, yerr=err, bottom=[yellow, blue, green, red], label='green')
# # ax.set_ylabel('Territory')
# # ax.set_xlabel('Sales')
# # ax.legend()  #显示图中左上角的标识区域

# # plt.show()

# # import numpy as np
# # import matplotlib.pyplot as plt
# # Emp_data= np.loadtxt('Employedpopulation.csv',delimiter = ",",
# #                      usecols=(1,2,3,4,5,6,7,8,9,10),dtype=int)

# # # 设置matplotlib正常显示中文和负号
# # plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
# # plt.rcParams['axes.unicode_minus']=False     # 正常显示负号

# #创建一个绘图对象, 并设置对象的宽度和高度
# plt.figure(figsize=(12, 4))
# #绘制全部就业人员柱状图
# plt.bar(am_1, am_4, width = 0.3, color = 'red')
# #绘制城镇就业人员柱状图
# plt.bar(ja_1, ja_4,width = 0.3, color = 'green')
# #绘制乡村就业人员柱状图
# plt.bar(ap_1, ap_4, width = 0.3, color = 'blue')
# plt.bar(em_1, em_4, width = 0.3, color = 'blue')

# # x = [i for i in labels]
# plt.xlabel('Sales')
# plt.ylabel('Territory')
# plt.ylim((30000,80000))
# # plt.xticks(x)
# # plt.title("2007-2016年城镇、乡村和全部就业人员情况柱状图")
# #添加图例
# plt.legend(('1','2','3','4'))
# plt.savefig('1.png')
# plt.show()



x = ['Americas', 'Japan', 'APAC', 'EMEA']
y1 = [am_1, ja_1, ap_1, em_1]
y2 = [am_2, ja_2, ap_2, em_2]
y3 = [am_3, ja_3, ap_3, em_3]
y4 = [am_4, ja_4, ap_4, em_4]

plt.bar(x, y4, label="Q1", color='red')
plt.bar(x, y2, label="Q2",color='orange')
plt.bar(x, y3, label="Q3", color='lightgreen')
plt.bar(x, y1, label="Q4", color='yellow')

plt.xticks(np.arange(len(x)), x, rotation=0, fontsize=10)  # 数量多可以采用270度，数量少可以采用340度，得到更好的视图
plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('Sales')
plt.xlabel('Territory')
plt.rcParams['savefig.dpi'] = 1600  # 图片像素
plt.rcParams['figure.dpi'] = 1600  # 分辨率
plt.rcParams['figure.figsize'] = (25.0, 15.0)  # 尺寸
plt.title("title")
plt.savefig('result.png')
plt.show()