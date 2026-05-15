import json
import os
ori_json = '/data/wanglinge/project/weighted-cav-mae/src/data_info/vgg/vggsound_train_cleaned.json'
cleaned_json = '/data/wanglinge/project/weighted-cav-mae/src/data_info/vgg/vggsound_train_cleaned.json'

data = []
with open(ori_json, 'r') as f:
    data = json.load(f)

num_frames = 16
def check_frames(root, id, num_frames=16):
    frames_idx = range(num_frames)
    frames_dirs = ["frames_{}".format(i) for i in frames_idx]
    png_name = "frame_{}.png".format(id)
    png_names = [os.path.join(root, frames_dir, png_name) for frames_dir in frames_dirs]
    for png in png_names:
        if not os.path.exists(png):
            return False
    return True



cleaned_data = {"data": []}
print("Total entries before cleaning:", len(data['data']))
for item in data['data']:
    root = item['video_path']
    if os.path.exists(item['raw_video_path'] and os.path.exists(item['wav'])) and check_frames():
        cleaned_data['data'].append(item)
print("Total entries after cleaning:", len(cleaned_data['data']))
with open(cleaned_json, 'w') as f:
    json.dump(cleaned_data, f, indent=4)