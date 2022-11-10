from module import *
from utils import get_positional_table, get_sinusoid_encoding_table
import hyperparams as hp
import copy

# 实现完成模型的构建


class Encoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, embedding_size, num_hidden):
        """
        :param embedding_size: dimension of embedding
        :param num_hidden: dimension of hidden
        """
        super(Encoder, self).__init__()
        # 使用正弦函数进行位置编码，并将其冻结
        self.alpha = nn.Parameter(t.ones(1))
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.encoder_prenet = EncoderPrenet(embedding_size, num_hidden)  # 实例化编码器侧预处理模块
        self.layers = clones(Attention(num_hidden), 3) # 3表示使用三层transformer block
        self.ffns = clones(FFN(num_hidden), 3)

    def forward(self, x, pos):
        '''
        :param x:可视为分词后的文本序列，[bsz,max_text_len]
        :param pos:文本序列对应的位置，[bsz,max_text_len]
        '''

        # Get character mask
        if self.training:
            c_mask = pos.ne(0).type(t.float) # [bsz,max_text_len]，padding部分为0，非padding部分为1
            # [bsz,max_text_len,max_text_len]，padding部分为1，非padding部分为0；因为module中使用mask_filled方法，是将1对应的位置值进行填充
            # 因为encoder计算的自注意力中key、value和query都是相等的，所以mask的后面两个维度大小是相等的
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)

        else:
            c_mask, mask = None, None

        # Encoder pre-network
        x = self.encoder_prenet(x) # 预处理

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x # 设置位置编码

        # Positional dropout
        x = self.pos_dropout(x)

        # Attention encoder-encoder
        attns = list() # 存放encoder中每一层自注意力计算的输出结果
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)

        return x, c_mask, attns # 此处x为整个encoder最后的输出


class MelDecoder(nn.Module):
    """
    Decoder Network
    """
    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden
        """
        super(MelDecoder, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.alpha = nn.Parameter(t.ones(1))
        self.decoder_prenet = Prenet(hp.num_mels, num_hidden * 2, num_hidden, p=0.2)
        self.norm = Linear(num_hidden, num_hidden)

        self.selfattn_layers = clones(Attention(num_hidden), 3)
        self.dotattn_layers = clones(Attention(num_hidden), 3)
        self.ffns = clones(FFN(num_hidden), 3)
        self.mel_linear = Linear(num_hidden, hp.num_mels * hp.outputs_per_step)
        self.stop_linear = Linear(num_hidden, 1, w_init='sigmoid')

        self.postconvnet = PostConvNet(num_hidden)

    def forward(self, memory, decoder_input, c_mask, pos):
        '''
        :param memory:encoder的输出，[bsz,max_text_len,num_hidden]
        :param decoder_input:decoder的输入，[bsz,max_T,num_mels]
        :param: c_mask:encoder输出对应的mask，[bsz,max_text_len]，对应memory
        :param: pos:[bsz,max_T]，对应decoder_input
        '''
        batch_size = memory.size(0)
        decoder_len = decoder_input.size(1)

        # get decoder mask with triangular matrix
        if self.training:
            m_mask = pos.ne(0).type(t.float) # [bsz,max_T]，padding部分为0，非padding部分为1
            mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
            if next(self.parameters()).is_cuda:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0) # [bsz,max_T,max_T]
            zero_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len) # [bsz,max_text_len,max_T]，padding部分为1，非padding部分为0
            zero_mask = zero_mask.transpose(1, 2) # [bsz,max_T,max_text_len]
        else:
            if next(self.parameters()).is_cuda:
                mask = t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0) # [bsz,max_T,max_T]
            m_mask, zero_mask = None, None

        # Decoder pre-network
        decoder_input = self.decoder_prenet(decoder_input)

        # Centered position
        decoder_input = self.norm(decoder_input)

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        decoder_input = pos * self.alpha + decoder_input

        # Positional dropout
        decoder_input = self.pos_dropout(decoder_input)

        # Attention decoder-decoder, encoder-decoder
        attn_dot_list = list() # 记录decoder中encoder和decoder的交叉注意力层的输出
        attn_dec_list = list() # 记录decoder中自注意力层的输出

        for selfattn, dotattn, ffn in zip(self.selfattn_layers, self.dotattn_layers, self.ffns):
            decoder_input, attn_dec = selfattn(decoder_input, decoder_input, mask=mask, query_mask=m_mask) # 自注意力
            decoder_input, attn_dot = dotattn(memory, decoder_input, mask=zero_mask, query_mask=m_mask) # 交叉注意力
            decoder_input = ffn(decoder_input)
            attn_dot_list.append(attn_dot)
            attn_dec_list.append(attn_dec)

        # Mel linear projection
        mel_out = self.mel_linear(decoder_input)
        
        # Post Mel Network
        postnet_input = mel_out.transpose(1, 2)
        out = self.postconvnet(postnet_input) # decoder侧的后处理
        out = postnet_input + out
        out = out.transpose(1, 2)

        # Stop tokens
        stop_tokens = self.stop_linear(decoder_input) # 停止符的预测

        return mel_out, out, attn_dot_list, stop_tokens, attn_dec_list


class Model(nn.Module):
    """
    Transformer Network
    """
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder(hp.embedding_size, hp.hidden_size)
        self.decoder = MelDecoder(hp.hidden_size)

    def forward(self, characters, mel_input, pos_text, pos_mel):
        memory, c_mask, attns_enc = self.encoder.forward(characters, pos=pos_text)
        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder.forward(memory, mel_input, c_mask,
                                                                                             pos=pos_mel)

        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec


class ModelPostNet(nn.Module):
    """
    CBHG Network (mel --> linear) 使用CBHG将mel谱图转换成显性的mag谱图
    """
    def __init__(self):
        super(ModelPostNet, self).__init__()
        self.pre_projection = Conv(hp.n_mels, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)
        self.post_projection = Conv(hp.hidden_size, (hp.n_fft // 2) + 1)

    def forward(self, mel):
        mel = mel.transpose(1, 2)
        mel = self.pre_projection(mel)
        mel = self.cbhg(mel).transpose(1, 2)
        mag_pred = self.post_projection(mel).transpose(1, 2)

        return mag_pred

if __name__ == '__main__':
    a = get_sinusoid_encoding_table(10, 5, padding_idx=[0, 5])
    print(a)