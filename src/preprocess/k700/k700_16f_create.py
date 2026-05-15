import json

def modify_json_paths(input_file, output_file, old_path, new_path):
    """
    修改JSON文件中的路径字符串
    
    参数:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        old_path: 需要替换的旧路径
        new_path: 替换后的新路径
    """
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据结构是否符合预期
        if 'data' not in data or not isinstance(data['data'], list):
            raise ValueError("JSON文件结构不符合预期，缺少'data'数组")
        
        # 遍历并修改路径
        for item in data['data']:
            # 修改video_path字段
            if 'video_path' in item and item['video_path'] == old_path:
                item['video_path'] = new_path
            
            # 如果需要，也可以修改wav路径（根据实际需求决定是否启用）
            # if 'wav' in item and old_path in item['wav']:
            #     item['wav'] = item['wav'].replace(old_path, new_path)
        
        # 写入修改后的JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"成功修改，结果已保存到 {output_file}")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    # 配置参数
    input_json = "/home/chenyingying/tmp/cav-mae-sync/src/data_info/k700/k700_val.json"    # 输入JSON文件名
    output_json = "/home/chenyingying/tmp/cav-mae-sync/src/data_info/k700/k700_val_16f.json"  # 输出JSON文件名
    old_path = "/data/wanglinge/dataset/data/k700/val"
    new_path = "/data/wanglinge/project/cav-mae/src/data/k700/val_16f"
    
    # 执行修改
    modify_json_paths(input_json, output_json, old_path, new_path)