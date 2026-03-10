# -*- coding: utf-8 -*-
"""NPZ 数据集制作示例：视频或 CSV 转成 NPZ（供 RNN/序列分类用）。"""

from BaseDT import npz

# 方式一：视频目录 → NPZ（目录结构：类别名/视频文件，如 video/挥手/*.mp4, video/走路/*.mp4）
npz.make_npz_dataset('./video', 'dataset_video.npz', data_type='video', sequence_length=30)

# 方式二：CSV 文件 → NPZ（最后一列为标签）
npz.make_npz_dataset('data.csv', 'dataset_csv.npz', data_type='csv', sequence_length=10, label_column=-1)

# 方式三：用类逐视频控制
# gen = npz.NPZGenerator(dataset_path='./video', sequence_length=30)
# gen.generate_dataset('out.npz')
