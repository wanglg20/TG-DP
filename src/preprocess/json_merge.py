import json
import argparse

def merge_json(file1, file2, output_file):
    # 读取第一个 JSON
    with open(file1, 'r') as f:
        data1 = json.load(f)
    
    # 读取第二个 JSON
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    # 合并 "data" 字段
    merged_data = {"data": data1["data"] + data2["data"]}
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":

    merge_json("/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/unbalanced_partial_partition.json", "/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/pending_valid.json", "/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/unbalanced_145w.json")

