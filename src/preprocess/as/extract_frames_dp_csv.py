import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import json
cv2.setNumThreads(0)

import os, math
import cv2
import numpy as np
from PIL import Image
from torchvision.utils import save_image

def extract_frame_robust(input_video_path: str, target_fold: str, extract_frame_num: int = 16, preprocess=None):
    """
    Robust frame extractor:
      - even sampling with linspace (inclusive endpoints)
      - frame-index seek -> msec seek fallback
      - warm-up read for unstable first frame
    """
    # —— Open video with FFMPEG backend if available
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    video_id = os.path.splitext(os.path.basename(input_video_path))[0]

    if not cap.isOpened():
        print(f'[{video_id}] ERROR: cannot open video')
        return 0

    # —— meta
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # 兜底：某些流返回 0/NaN
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0  # safe fallback
    if not frame_count or math.isnan(frame_count) or frame_count <= 0:
        # 强行估一个上限（比如 10 秒），至少让采样逻辑可运行
        frame_count = int(fps * 10)

    # 最多只取前 10 秒
    max_frames = int(min(frame_count, fps * 10))
    if max_frames <= 0:
        print(f'[{video_id}] ERROR: invalid frame_count after clamp')
        cap.release()
        return 0

    # —— positions to sample (inclusive of last)
    # e.g. extract_frame_num=10 -> indices like [0, ..., max_frames-1]
    if extract_frame_num == 1:
        sample_indices = np.array([0], dtype=int)
    else:
        sample_indices = np.linspace(0, max_frames - 1, num=extract_frame_num, dtype=int)
    sample_indices = np.unique(sample_indices)  # 去重，避免 very short clip

    saved = 0

    # —— warm-up: 对部分编码器，直接 seek 第 0 帧会失败
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, _ = cap.read()
    if not ok:
        # 再尝试重新打开+读一帧
        cap.release()
        cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
        cap.read()  # ignore result; just warm up

    for i, frame_idx in enumerate(sample_indices):
        # 1) 尝试按帧号定位
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        success, frame = cap.read()

        # 2) 回退：按时间戳定位（部分容器/编码器帧定位不精确）
        if not success:
            msec = (frame_idx / fps) * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, msec)
            success, frame = cap.read()

        # 3) 首帧特殊再次回退：有些视频第 0 帧就是坏的，试着读下一帧
        if not success and frame_idx == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            success, frame = cap.read()

        if not success or frame is None:
            print(f'[{video_id}] Warning: failed to read frame {frame_idx}')
            continue

        # —— save
        if preprocess is None:
            # 如果只是落盘做存档，用 cv2.imwrite 更快（BGR）
            # out_img = frame
            # cv2.imwrite(out_path, out_img)
            # 若你需要和原来一致的“预处理后张量保存”，维持下面流程
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            image_tensor = pil_to_tensor01(pil_im)  # 需要你实现或替换：转 0~1 tensor
        else:
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            image_tensor = preprocess(pil_im)

        frame_dir = os.path.join(target_fold, f'frame_{i}')
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir, exist_ok=True)
        out_path = os.path.join(frame_dir, f'{video_id}.jpg')
        save_image(image_tensor, out_path)
        saved += 1

        # 清理
        del frame

    cap.release()
    return saved

# 可选：如果你不需要 torchvision 的 preprocess，这里给个简单的替代
import torch
def pil_to_tensor01(pil_im):
    arr = np.array(pil_im)  # H W C, uint8
    ten = torch.from_numpy(arr).permute(2,0,1).float()/255.0
    return ten.clamp(0,1)



# preprocess pipeline
preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()
])

def extract_frame(input_video_path: str, target_fold: str, extract_frame_num: int = 10):
    """
    extract frames from video
    """
    ext = os.path.splitext(input_video_path)[1]
    video_id = os.path.basename(input_video_path).replace(ext, '')

    vidcap = cv2.VideoCapture(input_video_path)
    try:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))

        for i in range(extract_frame_num):
            frame_idx = int(i * (total_frame_num / extract_frame_num))
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
            success, frame = vidcap.read()
            if not success:
                #print(f'[{video_id}] Warning: failed to read frame {frame_idx}')
                continue

            # BGR → RGB → PIL → Tensor
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            image_tensor = preprocess(pil_im)

            frame_dir = os.path.join(target_fold, f'frame_{i}')
            if not os.path.exists(frame_dir):  # 避免每次都 makedirs
                os.makedirs(frame_dir, exist_ok=True)
            out_path = os.path.join(frame_dir, f'{video_id}.jpg')
            save_image(image_tensor, out_path)

            # 清理中间变量
            del cv2_im, pil_im, image_tensor, frame

    finally:
        vidcap.release()

def _worker(args):
    input_path, target_fold, num_frames = args
    try:
        extract_frame_robust(input_path, target_fold, num_frames)
    except Exception as e:
        print(f'[ERROR] {input_path}: {e}')

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract frames from videos")
    parser.add_argument(
        "-input_file_list", type=str,
        default='/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/audioset_20k_cleaned_converted.json',
        help="input file list"
    )
    parser.add_argument(
        "-target_fold", type=str,
        default='/data/wanglinge/dataset/AudioSet20k/frames',
        help="folder to save extracted frames"
    )
    parser.add_argument(
        "-num_workers", type=int, default=4,
        help="num of threads to use for parallel processing"
    )
    parser.add_argument(
        "-extract_frame_num", type=int, default=16,
        help="number of frames to extract from each video"
    )
    args = parser.parse_args()

    ori_data = json.load(open(args.input_file_list, 'r'))
    input_filelist = [item['raw_video_path'] for item in ori_data['data']]
    # ori_data = pd.read_csv(args.input_file_list, header=None)
    # input_filelist = ori_data.iloc[:, 0].tolist()


    tasks = [(path, args.target_fold, args.extract_frame_num) for path in input_filelist]
    n_workers = args.num_workers

    # 批次提交任务，减少内存积压
    batch_size = 100
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i+batch_size]
            futures = [executor.submit(_worker, task) for task in batch_tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing batch {i//batch_size+1}"):
                future.result()


            
    # print(f'[INFO] Total {len(tasks)} videos | Using {n_workers} workers')

    # # with Pool(processes=n_workers) as pool:
    # #     pool.map(_worker, tasks)

    # with Pool(processes=n_workers) as pool:
    #     results = []
    #     for _ in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc="Processing videos"):
    #         results.append(_)
    
    # print(f'[INFO] Done. {len([r for r in results if r is not None])} videos processed successfully.')



# Results:
#     📊 Statistics:
#    - Total input pairs: 542356
#    - Valid pairs: 467265
#    - Invalid/Missing pairs: 75091
#    - Success rate: 86.15%