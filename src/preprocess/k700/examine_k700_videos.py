import os 

data_info = '/data/wanglinge/dataset/OpenMMLab___Kinetics_700/raw/Kinetics_700/kinetics700_train_list_videos.txt'
video_list = []
root = '/data/wanglinge/dataset/OpenMMLab___Kinetics_700/raw/Kinetics_700/videos'
output_file = '/data/wanglinge/dataset/OpenMMLab___Kinetics_700/raw/Kinetics_700/kinetics700_train_valid_list_videos.txt'
label_list = []


valid = 0
invalid = 0
with open(data_info, 'r') as f:
    lines = f.readlines()
    for line in lines:
        video_path = line.strip().split(' ')[0]
        video_path_abs = os.path.join(root, video_path)
        if not os.path.exists(video_path_abs):
            invalid += 1
            print(f"Invalid video path: {video_path_abs}")
        else:
            video_list.append(video_path)
            label = line.strip().split(' ')[1]
            label_list.append(label)
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

with open(output_file, 'w') as f:
    for i, video in enumerate(video_list):
        video_path = os.path.join(root, video)
        label = label_list[i]
        f.write(f"{video_path} {label}\n")
print(f"Valid videos: {len(video_list)}, Invalid videos: {invalid}")
print(f"Output written to {output_file}")   