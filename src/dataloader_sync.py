# Portions of this file are adapted from the original CAV-MAE by Yuan Gong and David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os.path

import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL

def make_index_dict_k700(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            # index_lookup[row['mid']] = row['index']
            index_lookup[row['id']] = int(row['id'])
            line_count += 1
    return index_lookup

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        print(f"Original data length: {len(self.data)}")
        
        num_samples = audio_conf.get('num_samples')
        if num_samples is not None:
            n = min(num_samples, len(self.data))
            print(f"Limiting dataset to {n} samples (requested {num_samples})")
            self.data = self.data[:n]
        
        self.data = self.pro_data(self.data)
        self.num_samples = len(self.data)
        print(f'Dataset has {self.num_samples} samples after processing')
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.failed_audio_loadings = 0
        self.failed_image_loadings = 0
        self.debug_counter = 0  # Add a counter for debugging
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
            # fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
        else:
            print('not use noise augmentation')
        if self.dataset == 'k700':
            self.index_dict = make_index_dict_k700(label_csv)
        else:
            self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get('frame_use', -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get('total_frame', 16)
        print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))

        
        # Time-Shifting
        self.time_shifting = self.audio_conf.get('time_shifting', 0)



        # Aug
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

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['wav'], data_json[i]['labels'], data_json[i]['video_id'], data_json[i]['video_path']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        datum['video_path'] = np_data[3]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

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

    def randselect_img(self, video_id, video_path):
        if self.mode == 'eval':
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
            if self.mixup > 0:
                # Randomly select another frame
                while frame_idx == self.frame_use:
                    frame_idx = random.randint(0, self.total_frame - 1)

        else:
            frame_idx = random.randint(0, self.total_frame - 1)

        while os.path.exists(video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg') == False and frame_idx >= 1:
            print('frame {:s} {:d} does not exist'.format(video_id, frame_idx))
            frame_idx -= 1
        out_path = video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg'
        #print(out_path)
        return out_path, frame_idx

    def get_all_frames(self, video_id, video_path):
        all_frames = []
        for frame_idx in range(self.total_frame):
            frame_path = f"{video_path}/frame_{frame_idx}/{video_id}.jpg"
            if os.path.exists(frame_path):
                all_frames.append((frame_path, frame_idx))
        return all_frames

    def map_frame_to_spectrogram(self, frame_index, num_frames, spectrogram_length, target_length):
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

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset.

        This method handles both evaluation and training modes, processing audio and image data
        for the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            In evaluation mode:
                tuple: (fbanks, images, label_indices, video_id, frame_indices)
                    - fbanks (torch.Tensor): Stack of processed audio spectrograms.
                    - images (torch.Tensor): Stack of processed images.
                    - label_indices (torch.FloatTensor): One-hot encoded labels with smoothing.
                    - video_id (str): Identifier for the video.
                    - frame_indices (torch.Tensor): Indices of the processed frames.

            In training mode:
                tuple: (fbank, image, label_indices, video_id, frame_idx)
                    - fbank (torch.Tensor): Processed audio spectrogram.
                    - image (torch.Tensor): Processed image.
                    - label_indices (torch.FloatTensor): One-hot encoded labels with smoothing.
                    - video_id (str): Identifier for the video.
                    - frame_idx (int): Index of the processed frame.

        Note:
            In case of errors during processing, dummy data may be returned in training mode.
        """
        if index >= self.num_samples:
            raise IndexError(f"Index {index} is out of bounds for dataset with {self.num_samples} samples")
        
        datum = self.decode_data(self.data[index])
        if self.mixup > 0 and self.mode == 'eval':
            index2 = random.randint(0, self.num_samples - 1)
            datum2 = self.decode_data(self.data[index2])
        if self.mode == 'retrieval':
            fbanks = []
            images = []
            frame_indices = []
            
            for frame_idx in range(self.total_frame):
                frame_path = f"{datum['video_path']}/frame_{frame_idx}/{datum['video_id']}.jpg"
                
                try:
                    fbank = self._wav2fbank(datum['wav'])
                    # Use the mapping function to get the spectrogram segment
                    start, end = self.map_frame_to_spectrogram(
                        frame_index=frame_idx,
                        num_frames=self.total_frame,
                        spectrogram_length=fbank.shape[0],
                        target_length=self.target_length
                    )
                    fbank = fbank[start:end, :]
                    
                    if not self.skip_norm:
                        fbank = (fbank - self.norm_mean) / self.norm_std
                except Exception as e:
                    print(f"Error processing audio for video {datum['video_id']}: {str(e)}")
                    fbank = torch.zeros(self.target_length, 128)
                
                try:
                    image = self.get_image(frame_path)
                except Exception as e:
                    # Try to use previous frame if available, otherwise use zero tensor
                    image = (images[frame_idx - 1].clone() if frame_idx > 0 and images 
                            else torch.zeros(3, 224, 224))
                    self.failed_image_loadings += 1
                
                fbanks.append(fbank)
                images.append(image)
                frame_indices.append(frame_idx)
            
            if not fbanks:
                raise RuntimeError(f"No valid frames found for video {datum['video_id']}")
            
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            for label_str in datum['labels'].split(','):
                # label_str = label_str.strip('')
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)
            
            self.debug_counter += 1

            return torch.stack(fbanks), torch.stack(images), label_indices, datum['video_id'], torch.tensor(frame_indices)
        
        # else:  # Training mode
        elif self.mode == 'train':
            # Select a random frame
            frame_idx = random.randint(0, self.total_frame - 1)
        else:
            #frame_idx = int((self.total_frame) / 2)
            frame_idx = 5
            
        frame_path = f"{datum['video_path']}/frame_{frame_idx}/{datum['video_id']}.jpg"
            
        try:
            fbank = self._wav2fbank(datum['wav'])
            # Use the mapping function to get the spectrogram segment
            start, end = self.map_frame_to_spectrogram(
                frame_index=frame_idx,
                num_frames=self.total_frame,
                spectrogram_length=fbank.shape[0],
                target_length=self.target_length
            )
            fbank = fbank[start:end, :]
            time_len = fbank.shape[0]
            shift_step = int(self.time_shifting * time_len)
            if self.mode == 'train' and self.time_shifting > 0:
                shift = random.randint(-shift_step, shift_step)
                fbank = self.time_shift_spectrogram_circular(fbank, shift=shift, time_dim=0)



            image = self.get_image(frame_path)
            if self.mixup > 0:
                mix_frame_idx = random.randint(0, self.total_frame - 1)
                while mix_frame_idx == frame_idx:
                    mix_frame_idx = random.randint(0, self.total_frame - 1)
                frame_path2 = f"{datum2['video_path']}/frame_{mix_frame_idx}/{datum2['video_id']}.jpg"
                fbank2 = self._wav2fbank(datum2['wav'])
                start2, end2 = self.map_frame_to_spectrogram(
                    frame_index=mix_frame_idx,
                    num_frames=self.total_frame,
                    spectrogram_length=fbank2.shape[0],
                    target_length=self.target_length
                )
                fbank2 = fbank2[start2:end2, :] 
                fbank = fbank2 * self.mixup + fbank * (1 - self.mixup)

            if not self.skip_norm:
                fbank = (fbank - self.norm_mean) / self.norm_std
            
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)
            
            self.debug_counter += 1

            return fbank, image, label_indices, datum['video_id'], frame_idx
        except Exception as e:
            print(f"Error processing item at index {index}: {str(e)}")
            # Return dummy data in case of error
            return torch.zeros((self.target_length, 128)), torch.zeros((3, 224, 224)), torch.zeros(self.label_num), datum['video_id'], frame_idx

    def __len__(self):
        return self.num_samples
    
    def get_error_counts(self):
        return self.failed_audio_loadings, self.failed_image_loadings

    def reset_error_counts(self):
        self.failed_audio_loadings, self.failed_image_loadings = 0, 0

def eval_collate_fn(batch):
    fbanks, images, labels, video_ids, frame_indices = zip(*batch)
    
    fbanks = torch.cat(fbanks)
    images = torch.cat(images)
    labels = torch.stack(labels).repeat_interleave(fbanks.size(0) // len(labels), dim=0)
    video_ids = [vid for vid in video_ids for _ in range(len(frame_indices[0]))]
    frame_indices = torch.cat(frame_indices)
    
    return fbanks, images, labels, video_ids, frame_indices

# New function for training collate
def train_collate_fn(batch):
    fbanks, images, labels, video_ids, frame_indices = zip(*batch)
    
    fbanks = torch.stack(fbanks)
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    return fbanks, images, labels, video_ids, frame_indices


def test_audioset():
    audio_conf = {'num_mel_bins': 128, 'target_length': 416, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'dataset': 'as', 'mode':'train', 'mean':-5.081, 'std':4.4849,
                  'noise':False, 'label_smooth': 0, 'im_res': 224}
    
    dataset = AudiosetDataset('/home/chenyingying/tmp/cav-mae-sync/src/data_info/vgg/vggsound_test_cleaned.json', audio_conf, label_csv='/home/chenyingying/tmp/cav-mae-sync/src/data_info/vgg/class_labels_indices_vgg.csv')
    print(f"Dataset length: {len(dataset)}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=train_collate_fn)

    for i, (fbanks, images, labels, video_ids, frame_indices) in enumerate(loader):
        print(f"Batch {i}:")
        # print(f"  fbanks shape: {fbanks.shape}")
        # print(f"  images shape: {images.shape}")
        # print(f"  labels shape: {labels.shape}")
        print(frame_indices)
        if i > 10:
            break

        
if __name__ == '__main__':
    test_audioset()
    print("Test completed.")
    