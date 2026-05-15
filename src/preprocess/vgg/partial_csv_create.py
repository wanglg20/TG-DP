import json
import csv
import os

files_per_folder = 5000
ext = '.mp4'
output_file = '/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/unbalanced_partial_.csv'
data = []

video_root = '/data/wanglinge/dataset/Fast-Audioset-Download/wavs'
total_files = 0
total_missing = 0

# first csv:
csv_0 = '/data/wanglinge/dataset/Fast-Audioset-Download/processed_splits/unbalanced_train_segments_split_00.csv'
num_missing = 0
with open(csv_0, 'r') as f:
    file = f.read()

# file = open(f'cleaned_csvs/{split}.csv', 'r').read()
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

    video_path = os.path.join(video_root, 'unbalanced_train_segments_split_00', subfolder_idx, f'id_{ids}{ext}')
    if not os.path.exists(video_path):
        subfolder_idx = f'{0 // files_per_folder:06}'
        video_path = os.path.join(video_root, 'unbalanced_train_segments_split_00', subfolder_idx, f'id_{ids}{ext}')
    if not os.path.exists(video_path):
        num_missing += 1
        #print(f"Missing file: {video_path}")
        continue
    else:
        data.append({
            'video_path': video_path,
            'labels': labels
        })
print(f"Missing files in split unbalanced_train_segments_split_00: {num_missing}")
total_files += len(rows) - num_missing
total_missing += num_missing



# chunk download part
split_root = '/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/cleaned_csvs/split_chunks'
num_splits = 45
splits_idx = range(1, num_splits+1)
splits = [f'unbalanced_train_segments_split_{i:02}' for i in splits_idx]
# splits = [os.path.join(split_root, split) for split in splits]

for split in splits:
    split_path = os.path.join(split_root, f'{split}.csv')
    num_missing = 0
    with open(split_path, 'r') as f:
        file = f.read()
    # file = open(f'cleaned_csvs/{split}.csv', 'r').read()
    rows = file.split('\n')[3:-1]
    #print(f"Processing split: {split}")
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

        video_path = os.path.join(video_root, split, subfolder_idx, f'id_{ids}{ext}')
        if not os.path.exists(video_path):
            subfolder_idx = f'{0 // files_per_folder:06}'
            video_path = os.path.join(video_root, split, subfolder_idx, f'id_{ids}{ext}')
        if not os.path.exists(video_path):
            num_missing += 1
            #print(f"Missing file: {video_path}")
            continue
        else:
            data.append({
                'video_path': video_path,
                'labels': labels
            })
    print(f"Missing files in split {split}: {num_missing}")
    total_missing += num_missing
    total_files += len(rows) - num_missing



# Write to CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['video_path', ' ', 'labels']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for entry in data:
        writer.writerow(entry)

print(f"Total files processed: {total_files}")
print(f"Total missing files: {total_missing}")