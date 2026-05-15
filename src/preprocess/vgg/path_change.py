import csv
import numpy as np
path = '/data/wanglinge/project/cav-mae/src/data/info/k700_val.csv'
# change root path /mnt to /data
output_path = '/data/wanglinge/project/cav-mae/src/data/info/k700_val.csv'
videos_list = np.loadtxt(path, dtype=str, delimiter=',')
with open(output_path, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for video in videos_list:
        video = video.replace('/mnt/', '/data/')
        writer.writerow([video])
    print(f"转换完成，输出文件路径：{output_path}")
