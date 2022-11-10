import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse

# 加载保存的模型参数
def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def synthesis(text, args):
    m = Model()
    m_post = ModelPostNet()
    # 模型加载参数
    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    # 因为文本分析序列化
    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = t.LongTensor(text).unsqueeze(0) # 需要是一个二维张量
    text = text.cuda()
    mel_input = t.zeros([1,1, 80]).cuda()
    pos_text = t.arange(1, text.size(1)+1).unsqueeze(0) # 用于构建mask
    pos_text = pos_text.cuda()

    m=m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)
    
    pbar = tqdm(range(args.max_len)) # 使用max_len设置最大的预测长度
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            mel_input = t.cat([mel_input, mel_pred[:,-1:,:]], dim=1) # 将每次预测生成的mel与之前的mel进行cat作为一下次计算的输入

        mag_pred = m_post.forward(postnet_pred) # 使用最后输出的经过postconvnet处理的mel谱图生成mag谱图
        
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy()) # 基于mag谱图生成音频
    write(hp.sample_path + "/test.wav", hp.sr, wav)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=160000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=100000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=400)

    args = parser.parse_args()
    synthesis("the 'lower-case' being in fact invented in the early Middle Ages.",args)
