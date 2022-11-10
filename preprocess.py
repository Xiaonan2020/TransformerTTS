import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal
import torch as t
import math
# 模型训练数据的加载与相关预处理

# 创建模型训练所使用的数据集
class LJDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.wav' # idx对应的音频文件路径
        text = self.landmarks_frame.iloc[idx, 1] # idx对应的文本内容

        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        # 将英文文本转为序列，相当于字符级别的分词，在最后都会加上一个1

        mel = np.load(wav_name[:-4] + '.pt.npy') # 加载梅尔谱图

        # 将[[0] * 80]与mel中的前n-1行在垂直方向concat，即去掉mel的最后一行，并且在最前面添加全为0的一行，作为输入
        mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text) # 序列长度
        pos_text = np.arange(1, text_length + 1) # 位置编码？？？
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text}

        return sample

# 用于后续加载mel图谱和mag谱图数据
class PostDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.wav'
        mel = np.load(wav_name[:-4] + '.pt.npy')
        mag = np.load(wav_name[:-4] + '.mag.npy')
        sample = {'mel':mel, 'mag':mag}

        return sample

# 用于对LJDatasets类构建的数据进行batch中的转换处理
def collate_fn_transformer(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        text = [d['text'] for d in batch] # batch中所有的文本数据
        mel = [d['mel'] for d in batch]  # batch中所有的mel数据
        mel_input = [d['mel_input'] for d in batch] # batch中所有的mel_input
        text_length = [d['text_length'] for d in batch] # batch中所有的test_length
        pos_mel = [d['pos_mel'] for d in batch] # batch中所有的pos_mel
        pos_text= [d['pos_text'] for d in batch] # batch中所有的pos_text

        # 将每个text与其对应的长度text_length匹配，以长度为标准对text进行降序排序，最后的列表中只取text
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        # 将每个melt与其对应的长度text_length匹配，以长度为标准对mel进行降序排序，最后的列表中只取mel
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        # 下面几项也是如此，就是以text_length的大小进行降序排序
        mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)# 用0将text中的每个文本序列都pad到最长的文本序列的长度
        mel = _pad_mel(mel)# 对mel进行pad
        mel_input = _pad_mel(mel_input)# 对mel_input进行pad
        pos_mel = _prepare_data(pos_mel).astype(np.int32)# 用0将pos_mel中的每个序列都pad到最长的序列的长度
        pos_text = _prepare_data(pos_text).astype(np.int32)# 用0将pos_text中的每个序列都pad到最长的序列的长度


        return t.LongTensor(text), t.FloatTensor(mel), t.FloatTensor(mel_input), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

# 用于对PostDatasets类构建的数据进行batch中的转换处理
def collate_fn_postnet(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]
        
        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return t.FloatTensor(mel), t.FloatTensor(mag)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

# 使用0对输出的x进行pad到指定长度length
def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

# 将inputs中所有的序列用0pad到其中最长序列的长度
def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

# 将一个batch中所有的mel用0pad到其中最大长度的大小
def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

# 计算模型的参数大小
def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def get_dataset():
    return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def get_post_dataset():
    return PostDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))


def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)