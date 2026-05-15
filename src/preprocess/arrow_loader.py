# -*- coding: utf-8 -*-
import os
import io
import random

from collections import OrderedDict
from typing import Any, Dict, Tuple, List, Optional

import wave

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from bisect import bisect_right
from torch.utils.data import Dataset

from PIL import Image
import PIL
import numpy as np
import soundfile as sf  # pip install soundfile

from dataloader_sync import make_index_dict, make_index_dict_k700, make_name_dict, lookup_list
import torchvision.transforms as T
import torch
import numpy as np

def _open_arrow_file(shard_path: str) -> ipc.RecordBatchFileReader:
    source = pa.memory_map(shard_path, "r")
    #reader = ipc.open_file(source)
    return ipc.open_file(source), source

class ArrowAVDataset(Dataset):
    """
    通过 manifest.parquet 定位到 (shard_path, row_in_shard)，
    将每个分片 mmap 到内存，只在第一次访问该分片时打开。
    """
    def __init__(self, audio_conf, manifest_path: str, label_csv: str,
                 transform_img=None, transform_audio=None, shuffle_index: bool = False):
        self.manifest = pq.read_table(manifest_path).to_pydict()
        self.n = len(self.manifest["global_index"])
        self.indices = list(range(self.n))
        if shuffle_index:
            random.shuffle(self.indices)

        # original audio conf:
        self.audio_conf = audio_conf
        # basical params
        self.dataset      = self.audio_conf.get("dataset", "as")
        self.sample_rate  = self.audio_conf.get("sample_rate", 16000)
        self.melbins      = self.audio_conf.get("num_mel_bins", 128)
        self.norm_mean    = torch.tensor(self.audio_conf.get("mean", 0.0))
        self.norm_std     = torch.tensor(self.audio_conf.get("std", 1.0))
        self.mode         = self.audio_conf.get("mode", "train")
        self.total_frame  = self.audio_conf.get("total_frame", 16)
        self.frame_use    = self.audio_conf.get("frame_use", -1)
        self.im_res       = self.audio_conf.get("im_res", 224)
        self.label_smooth = self.audio_conf.get("label_smooth", 0.0)
        self.target_length = self.audio_conf.get("target_length", 1024)  
        self.skip_norm     = self.audio_conf.get("skip_norm", False)
        # Time-Shifting
        self.time_shifting = self.audio_conf.get('time_shifting', 0)
        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))
        
        self.augmentation = self.audio_conf.get('augmentation', False)
        if self.augmentation:
            self.preprocess = T.Compose([
                T.RandomResizedCrop(self.im_res, scale=(0.08, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.4850, 0.4560, 0.4060],
                    std=[0.2290, 0.2240, 0.2250]
                )])
        else:
            self.preprocess = T.Compose([
                T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
                T.CenterCrop(self.im_res),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.4850, 0.4560, 0.4060],
                    std=[0.2290, 0.2240, 0.2250]
                )])
        if self.dataset == 'k700':
            self.index_dict = make_index_dict_k700(label_csv)
        else:
            self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        # cache: shard_path -> {"reader": RecordBatchFileReader, "row_offsets": List[int], "batch_cache": OrderedDict[int, pa.RecordBatch]}
        self._max_cache_size = self.audio_conf.get("max_shard_cache", 4)
        self._max_batch_cache = self.audio_conf.get("max_batch_cache", 1)
        self._shard_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        self.transform_img = transform_img
        self.transform_audio = transform_audio
    
    def __len__(self):
        return self.n

    def _get_row(self, i: int) -> Tuple[pa.RecordBatch, int]:
        j = self.indices[i]
        shard = self.manifest["shard_path"][j]
        rowid = int(self.manifest["row_in_shard"][j])

        entry = self._get_shard_entry(shard)
        row_offsets = self._ensure_row_offsets(entry)
        batch_idx = bisect_right(row_offsets, rowid)
        prev_total = row_offsets[batch_idx - 1] if batch_idx > 0 else 0
        row_in_batch = rowid - prev_total       # rowid in our setting
        batch = self._get_record_batch(entry, batch_idx)
        return batch, row_in_batch

    def _get_column_scalar(self, batch: pa.RecordBatch, rowid: int, name: str) -> pa.Scalar:
        col_index = batch.schema.get_field_index(name)
        if col_index == -1:
            raise KeyError(f"Column '{name}' not found in table")
        column = batch.column(col_index)
        return column[rowid]

    def _get_shard_entry(self, shard: str) -> Dict[str, Any]:
        entry = self._shard_cache.get(shard)
        if entry is not None:
            self._shard_cache.move_to_end(shard)
            return entry

        reader = _open_arrow_file(shard)
        entry = {
            "reader": reader,
            "row_offsets": None,
            "batch_cache": OrderedDict(),
        }
        self._add_to_cache(shard, entry)
        return entry

    def _add_to_cache(self, shard: str, entry: Dict[str, Any]) -> None:
        while len(self._shard_cache) >= self._max_cache_size:
            old_shard, old_entry = self._shard_cache.popitem(last=False)
            self._close_shard_entry(old_entry)
        self._shard_cache[shard] = entry

    def _ensure_row_offsets(self, entry: Dict[str, Any]) -> List[int]:
        if entry["row_offsets"] is not None:
            return entry["row_offsets"]

        reader, _ = entry["reader"]
        cumulative: List[int] = []
        total = 0
        for batch_idx in range(reader.num_record_batches):
            batch = reader.get_batch(batch_idx)
            total += batch.num_rows
            cumulative.append(total)
        entry["row_offsets"] = cumulative
        return cumulative

    def _get_record_batch(self, entry: Dict[str, Any], batch_idx: int) -> pa.RecordBatch:
        batch_cache: "OrderedDict[int, pa.RecordBatch]" = entry["batch_cache"]
        batch = batch_cache.get(batch_idx)
        if batch is not None:
            batch_cache.move_to_end(batch_idx)
            return batch

        reader, _ = entry["reader"]
        batch = reader.get_batch(batch_idx)
        batch_cache[batch_idx] = batch
        while len(batch_cache) > self._max_batch_cache:
            batch_cache.popitem(last=False)
        return batch

    def _close_shard_entry(self, entry: Dict[str, Any]) -> None:
        batch_cache: "OrderedDict[int, pa.RecordBatch]" = entry.get("batch_cache", OrderedDict())
        batch_cache.clear()
        reader, source= entry.get("reader")
        if source is not None:
            source.close()

    def _decode_frames_from_bytes(self, frames: List[bytes]) -> List[Image.Image]:
        imgs = []
        for b in frames:
            im = Image.open(io.BytesIO(b)).convert("RGB")
            imgs.append(im)
        return imgs

    def _decode_single_frame_from_buffer(self, buffer: pa.Buffer) -> Image.Image:
        # BufferReader avoids instantiating intermediate Python bytes objects.
        with pa.BufferReader(buffer) as reader:
            with Image.open(reader) as im:
                return im.convert("RGB")

    def _decode_audio_from_bytes(self, b: Any) -> Tuple[np.ndarray, int]:
        if isinstance(b, pa.Buffer):
            byte_source = memoryview(b)
        elif isinstance(b, (bytes, bytearray, memoryview)):
            byte_source = b
        else:
            byte_source = bytes(b)
        with io.BytesIO(byte_source) as bio:
            data, sr = sf.read(bio, dtype="float32", always_2d=False)
        return data, sr

    def _load_frames_from_paths(self, paths: List[str]) -> List[Image.Image]:
        imgs = []
        for p in paths:
            with open(p, "rb") as f:
                im = Image.open(f).convert("RGB")
            imgs.append(im)
        return imgs

    def _load_audio_from_path(self, p: str) -> Tuple[np.ndarray, int]:
        data, sr = sf.read(p, dtype="float32", always_2d=False)
        return data, sr

    def _wav2fbank(self, waveform: torch.Tensor, sr: int):
        waveform = waveform - waveform.mean()
        try:
            import torchaudio
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10
            )
        except Exception:
            fbank = torch.zeros([512, self.melbins]) + 0.01

        # # pad/crop 到 target_length
        # n = fbank.size(0)
        # if n < self.target_length:
        #     pad = self.target_length - n
        #     fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad))
        # elif n > self.target_length:
        #     fbank = fbank[:self.target_length, :]
        # return fbank

        target_length = 1024
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def _preprocess_img(self, img: Image.Image) -> torch.Tensor:
        image_tensor = self.preprocess(img)
        return image_tensor

    def _map_frame_to_window(self, frame_index, num_frames, spectrogram_length, target_length):
        """
        Maps a frame index to a corresponding segment in the spectrogram.
        
        :param frame_index: Index of the frame (0 to num_frames-1)
        :param num_frames: Total number of frames
        :param spectrogram_length: Total length of the spectrogram
        :param target_length: Desired length of each segment
        :return: Tuple of (start_index, end_index) for the spectrogram segment
        """
        frame_position = int(round(frame_index * spectrogram_length / num_frames))
        
        start = max(0, frame_position - target_length // 2)
        end = start + target_length
        
        if end > spectrogram_length:
            end = spectrogram_length
            start = max(0, end - target_length)
        
        return (start, end)
    
    def time_shift_spectrogram_circular(self, S: torch.Tensor, shift: int, time_dim: int = 0):
        """
        S: [T, F] 或更高维，但 time_dim 表示时间维
        shift: 正为向右滚动（后移），负为向左滚动（前移）
        """
        return torch.roll(S, shifts=shift, dims=time_dim)

    def __getitem__(self, i: int):
        batch, rowid = self._get_row(i)
        store = "bytes"
        if "store" in batch.schema.names:
            store_scalar = self._get_column_scalar(batch, rowid, "store")
            if store_scalar.is_valid:
                store = store_scalar.as_py()
        T_total = self.total_frame
        if self.mode == "train" and self.frame_use < 0:
            frame_idx = random.randint(0, max(0, T_total - 1))
        elif self.frame_use >= 0:
            frame_idx = min(self.frame_use, T_total - 1)
        else:
            frame_idx = 5

        frames_scalar = self._get_column_scalar(batch, rowid, "frames")
        if not frames_scalar.is_valid:
            raise ValueError("Invalid frame data")
        if store == "path":
            frames = frames_scalar.as_py()
            if not frames:
                raise ValueError("Empty frame list")
            chosen_frame_idx = min(frame_idx, len(frames) - 1)
            with open(frames[chosen_frame_idx], "rb") as f:
                with Image.open(f) as im:
                    img = im.convert("RGB")
        else:
            # available_frames = frames_scalar.value_length
            # if available_frames == 0:
            #     raise ValueError("Empty frame list")
            available_frames = len(frames_scalar)
            chosen_frame_idx = min(frame_idx, available_frames - 1)
            #frame_value_index = frames_scalar.offset + chosen_frame_idx
            frame_scalar = frames_scalar[chosen_frame_idx]
            frame_buffer = frame_scalar.as_buffer()
            img = self._decode_single_frame_from_buffer(frame_buffer)
        frame_idx = chosen_frame_idx
        img = self._preprocess_img(img)
        audio_scalar = self._get_column_scalar(batch, rowid, "audio")
        if not audio_scalar.is_valid:
            raise ValueError("Invalid audio data")
        audio_buffer = audio_scalar.as_buffer()
        wav, sr = self._decode_audio_from_bytes(audio_buffer)
        wav = np.array(wav, dtype=np.float32)
        if wav.ndim == 1:                       # channel = 1
            wav = np.expand_dims(wav, axis=1)  # [1, T]
        wav = torch.from_numpy(wav).transpose(0, 1).contiguous()  # [C,T]
        fbank_full = self._wav2fbank(wav, sr)                # [N, mel]
        start, end = self._map_frame_to_window(
            frame_index=frame_idx,
            num_frames=T_total,
            spectrogram_length=fbank_full.size(0),
            target_length=self.target_length,
        )
        fbank = fbank_full[start:end, :]
        time_len = fbank.size(0)
        shift_step = int(self.time_shifting * time_len)
        if self.mode == 'train' and self.time_shifting > 0:
            shift = random.randint(-shift_step, shift_step)
            fbank = self.time_shift_spectrogram_circular(fbank, shift=shift, time_dim=0)

        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std + 1e-8)

        labels_scalar = self._get_column_scalar(batch, rowid, "labels")
        labels = labels_scalar.as_py() if labels_scalar.is_valid else []
        key_scalar = self._get_column_scalar(batch, rowid, "key")
        key = key_scalar.as_py() if key_scalar.is_valid else None
        label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
        for label_str in labels:
            label_str = label_str.strip('"')
            label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
        label_indices = torch.FloatTensor(label_indices)

        return fbank, img, label_indices, key, frame_idx

def train_collate_fn(batch):
    fbanks, images, labels, video_ids, frame_indices = zip(*batch)
    
    fbanks = torch.stack(fbanks)
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    return fbanks, images, labels, video_ids, frame_indices

if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/data/wanglinge/dataset/VGGSound/arrow/train/manifest.parquet", help="manifest.parquet 路径")
    parser.add_argument("--limit", type=int, default=0, help="仅取前 N 条样本，0 则不限制")
    args = parser.parse_args()

    audio_conf = {'num_mel_bins': 128, 'target_length': 416, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'dataset': 'as', 'mode':'train', 'mean':-5.081, 'std':4.4849,
                  'noise':False, 'label_smooth': 0, 'im_res': 224}
    audio_conf = {'num_mel_bins': 128, 'target_length': 416, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'dataset': 'vggsound', 'mode': 'train', 'mean': -5.081, 'std': 4.4849,
                   'noise': True, 'label_smooth': 0, 'im_res': 224, 'shuffle': True}
    label_csv = '/data/wanglinge/project/weighted-cav-mae/src/data_info/vgg/class_labels_indices_vgg.csv'
    ds = ArrowAVDataset(audio_conf, args.manifest, label_csv)
    if args.limit > 0:
        ds.indices = ds.indices[:args.limit]
    print(f"数据集大小: {len(ds)}")

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=train_collate_fn)

    for i, (fbanks, images, labels, video_ids, frame_indices) in enumerate(loader):
        print(f"Batch {i}:")
        # print(f"  fbanks shape: {fbanks.shape}")
        # print(f"  images shape: {images.shape}")
        print(f"  labels shape: {labels.shape}")
        # print(frame_indices)
        if i > 10:
            break