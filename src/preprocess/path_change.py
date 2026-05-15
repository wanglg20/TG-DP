import json
import os

in_file = "/data/home/zdhs0059/wanglinge/project/weighted-cav-mae/src/data_info/AS2M/unbalanced_145w.json"       # 输入的 json 文件
out_file = "/data/home/zdhs0059/wanglinge/project/weighted-cav-mae/src/data_info/AS2M/unbalanced_145w_valid.json"  # 输出的 json 文件

# 替换前后的前缀
old_video_prefix = "/data/wanglinge/dataset/AudioSet2M/frames"
old_wav_prefix = "/data/wanglinge/dataset/AudioSet2M/audio"

new_video_prefix = "/data/home/zdhs0059/wanglinge/dataset/VGG_test/frames_16/test/"
new_wav_prefix = "/data/home/zdhs0059/wanglinge/dataset/AS2M/raw_data/frames"

with open(in_file, "r") as f:
    data = json.load(f)

for item in data["data"]:
    if "video_path" in item and item["video_path"].startswith(old_video_prefix):
        item["video_path"] = item["video_path"].replace(old_video_prefix, new_video_prefix, 1)
    if "wav" in item and item["wav"].startswith(old_wav_prefix):
        item["wav"] = item["wav"].replace(old_wav_prefix, new_wav_prefix, 1)

with open(out_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"修改完成，保存到 {out_file}")
