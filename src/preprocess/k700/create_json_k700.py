import os
import json
import csv
from collections import defaultdict


def generate_json_manifest(csv_path, output_json, audio_base_dir, video_frame_base_dir):
    """
    参数：
    csv_path: 输入CSV文件路径（格式：video_path, label）
    output_json: 输出JSON文件路径
    audio_base_dir: 音频文件存储根目录
    video_frame_base_dir: 视频帧存储根目录
    """
    # 读取CSV数据并聚合标签
    label_mapping = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            if len(header) != 2:
                raise ValueError("CSV格式错误：必须包含两列（视频路径和标签）")
        except StopIteration:
            raise ValueError("CSV文件为空或格式不正确")

        for row_num, row in enumerate(reader, 2):  # 从第2行开始计数
            if len(row) < 2:
                print(f"警告：跳过第{row_num}行，数据列不足")
                continue
            
            video_path, label = row[0], row[1]
            
            # 校验视频路径
            if not os.path.exists(video_path):
                print(f"警告：跳过不存在的视频文件 {video_path}")
                continue
                
            # 提取video_id
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            
            # 存储标签（自动去重）
            if label.strip() and label not in label_mapping[video_id]:
                label_mapping[video_id].append(label.strip())

    # 构建JSON数据结构
    json_data = {"data": []}
    for video_id, labels in label_mapping.items():
        # 构建音频路径
        audio_path = os.path.join(
            audio_base_dir,
            f"{video_id}.wav"
        )
        
        # 构建视频帧路径（根据实际存储结构调整）
        # 示例格式：/base/path/--4gqARaEJE/frame_0.jpg
        # 这里保持与问题中的示例一致，使用统一目录
        frame_dir = video_frame_base_dir
        
        entry = {
            "video_id": video_id,
            "wav": audio_path,
            "video_path": frame_dir,
            "labels": ",".join(sorted(labels))  # 排序保证一致性
        }
        json_data["data"].append(entry)

    # 写入JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"成功生成清单文件：{output_json}，包含 {len(json_data['data'])} 个条目")

if __name__ == "__main__":
    # 配置参数
    CSV_PATH =  '/data/wanglinge/project/cav-mae/src/data/info/k700_train_with_label.csv'  # 输入CSV文件路径
    OUTPUT_JSON = '/data/wanglinge/project/cav-mae/src/data/info/k700/k700_train_16f.json'  # 输出JSON路径
    AUDIO_BASE_DIR = "/data/wanglinge/project/cav-mae/src/data/k700/train/audio"  # 音频文件根目录
    VIDEO_FRAME_DIR = "/data/wanglinge/project/cav-mae/src/data/k700/train_16f"  # 视频帧根目录

    # 执行生成
    try:
        generate_json_manifest(CSV_PATH, OUTPUT_JSON, AUDIO_BASE_DIR, VIDEO_FRAME_DIR)
    except Exception as e:
        print(f"生成失败：{str(e)}")
