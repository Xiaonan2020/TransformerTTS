import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from utils import get_spectrograms
import hyperparams as hp
import librosa

# 定义一个数据类型，从LJSpeech数据集中提取训练所需的文本、语音的mel谱图和mag特征/显性图谱。
# 该文件主要是从.wav文件中提取出所需的特征数据并进行保存，方便后续使用时调用

# 主要用于从wave文件中提取特征数据
class PrepareDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. 文本数据的路径
            root_dir (string): Directory with all the wavs. 音频数据的路径

        """
        # 文本数据中每一行为一个数据，前面为对应音频的文件名，后面为具体的文本内容，与音频中的内容一致
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None) # 该csv文件以'|'划分
        # (13100, 3) ['LJ001-0008' 'has never been surpassed.' 'has never been surpassed.']
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame) # 返回数据量

    def __getitem__(self, idx): # 通过索引获取对应的文本和音频的mel图谱
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.loc[idx, 0]) + '.wav'
        mel, mag = get_spectrograms(wav_name) # 提取音频的mel和mag

        # 将从wav音频文件中提取的mel和mag保存至data中，方便后续训练使用;
        # 每个mel图谱的尺寸是[n, 80],每个mag图谱的尺寸是[n, 1025]，因为不同音频的长度是不一样的，故n的大小是不一致的
        np.save(wav_name[:-4] + '.pt', mel)
        np.save(wav_name[:-4] + '.mag', mag)

        sample = {'mel':mel, 'mag': mag}

        return sample
    
if __name__ == '__main__':
    dataset = PrepareDataset(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))
    # dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=1)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
