# delete invalid video-audio pairs 
import os
import json
import torchaudio
import math
from tqdm import tqdm
input_json_file = '/data/wanglinge/project/cav-mae/src/data/info/k700/k700_train.json'
output_json_file = '/data/wanglinge/project/cav-mae/src/data/info/k700/k700_train_valid.json'



valid_json_data = {"data": []}

with open(input_json_file, 'r') as f:
    data = json.load(f)['data']
    print(f"Total items in input JSON: {len(data)}")
    
    # 添加进度条
    valid_count = 0
    for item in tqdm(data, desc="Processing audio-video pairs", unit="item"):
        video_path = item['video_path']
        audio_path = item['wav']
        label = item['labels']
        video_id = item['video_id']
        if os.path.exists(video_path) and os.path.exists(audio_path):
            try:
                waveform, sr = torchaudio.load(audio_path)
                delta = 160000 - waveform.shape[1]
                if abs(delta) < 5000:
                    valid_json_data['data'].append({
                        'video_path': video_path,
                        'wav': audio_path,
                        'labels': label,
                        'video_id': video_id
                    })
                    valid_count += 1
            except Exception as e:
                # 如果音频文件损坏或无法读取，跳过此项
                # print(f"\nWarning: Failed to load audio {audio_path}: {e}")
                continue
        
with open(output_json_file, 'w') as f:
    json.dump(valid_json_data, f, indent=4)

print(f"\n✅ Processing completed!")
print(f"📁 Valid audio-video pairs saved to: {output_json_file}")
print(f"📊 Statistics:")
print(f"   - Total input pairs: {len(data)}")
print(f"   - Valid pairs: {len(valid_json_data['data'])}")
print(f"   - Invalid/Missing pairs: {len(data) - len(valid_json_data['data'])}")
print(f"   - Success rate: {len(valid_json_data['data'])/len(data)*100:.2f}%")

