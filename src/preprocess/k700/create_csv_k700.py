import os 
import csv

data_info = '/data/wanglinge/dataset/OpenMMLab___Kinetics_700/raw/Kinetics_700/kinetics700_train_list_videos.txt'
video_list = []
root = '/data/wanglinge/dataset/OpenMMLab___Kinetics_700/raw/Kinetics_700/videos'
output_file = '/data/wanglinge/project/cav-mae/src/data/info/k700/k700_train.csv'
label_list = []

with open(data_info, 'r') as f:
    lines = f.readlines()
    for line in lines:
        video_path = line.strip().split(' ')[0]
        label = line.strip().split(' ')[1]
        video_list.append(video_path)
        label_list.append(label)

# 使用csv模块写入两列数据
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # # 写入列标题
    # writer.writerow(['path', 'label'])
    
    # 写入数据行
    for i, video in enumerate(video_list):
        video_path = os.path.join(root, video)
        label = label_list[i]
        writer.writerow([video_path, label])