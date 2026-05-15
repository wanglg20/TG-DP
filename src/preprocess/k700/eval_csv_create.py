import json
import csv
import os


files_per_folder = 5000
ext = '.mp4'
output_file = '/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/eval.csv'
data = []

video_root = '/data/wanglinge/dataset/Fast-Audioset-Download/wavs/eval'
total_files = 0
num_missing = 0
original_csv = '/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/cleaned_csvs/cleaned_eval_segments.csv'
with open(original_csv, 'r') as f:
    file = f.read()

subfolder_list = ['000000', '000001', '000002', '000003', '000004']
rows = file.split('\n')[3:-1]
for i, row in enumerate(rows):
    video_info = row.replace(' ', '').split(',')
    ids = video_info[0]
    to = float((video_info[2]))
    start = float(video_info[1])
    labels = [c.replace('"','') for c in video_info[3:]]
    labels = ','.join(labels)
    subfolder_idx = f'{i // files_per_folder:06}'
    st = f'{int(start//3600)}:{int(start//60)-60*int(start//3600)}:{start%60}'
    dur = f'{int(to//3600)}:{int(to//60)-60*int(to//3600)}:{to%60}'

    video_path = os.path.join(video_root,  subfolder_idx, f'id_{ids}{ext}')

    if not os.path.exists(video_path):
        for subfolder in subfolder_list:
            video_path = os.path.join(video_root, subfolder, f'id_{ids}{ext}')
            if os.path.exists(video_path):
                break
    if not os.path.exists(video_path):
        num_missing += 1
        #print(f"Missing file: {video_path}")
        continue
    else:
        data.append({
            'video_path': video_path,
            'labels': labels
        })

print(f"Missing files in split eval: {num_missing}")
total_files += len(rows) - num_missing



os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['video_path', ' ', 'labels']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for entry in data:
        writer.writerow(entry)

print(f"Total files processed: {total_files}")