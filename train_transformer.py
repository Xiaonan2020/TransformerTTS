from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm

# 用于训练自回归注意网络，(text --> mel)

# 动态调整学习率
def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def main():

    dataset = get_dataset() # 获得数据
    global_step = 0
    
    m = nn.DataParallel(Model().cuda()) # 初始化模型；如果有多个gpu，在多个gpu上并行训练

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    pos_weight = t.FloatTensor([5.]).cuda()
    writer = SummaryWriter() # 初始化tensorboard中的对象
    
    for epoch in range(hp.epochs): # 训练10000个epoch

        # dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            global_step += 1 # 每训练一个batch就加1
            if global_step < 400000: # 当global_step小于400000之前，每个batch训练时都进行lr调整
                adjust_learning_rate(optimizer, global_step)
                
            character, mel, mel_input, pos_text, pos_mel, _ = data
            
            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)

            # 将数据都传入gpu中
            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()

            # 模型的前向计算
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)

            mel_loss = nn.L1Loss()(mel_pred, mel) # 未经postconvnet处理的mel的损失
            post_mel_loss = nn.L1Loss()(postnet_pred, mel) # 经过postconvnet处理的mel的损失
            
            loss = mel_loss + post_mel_loss
            
            writer.add_scalars('training_loss',{
                    'mel_loss':mel_loss,
                    'post_mel_loss':post_mel_loss,

                }, global_step) # 记录训练过程中的损失
                
            writer.add_scalars('alphas',{
                    'encoder_alpha':m.module.encoder.alpha.data,
                    'decoder_alpha':m.module.decoder.alpha.data,
                }, global_step) # 记录训练时位置编码中参数alpha

            
            
            if global_step % hp.image_step == 1: # 每训练500个batch
                
                for i, prob in enumerate(attn_probs): # 将decoder中的交叉注意力保存为图像
                    print(prob.shape)

                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*4] * 255)
                        writer.add_image('Attention_%d_0'%global_step, x, i*4+j)
                
                for i, prob in enumerate(attns_enc): # 将encoder中的自注意力保存为图像
                    num_h = prob.size(0)
                    
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*4] * 255)
                        writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)
            
                for i, prob in enumerate(attns_dec): # 将decoder中的自注意力保存为图像

                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*4] * 255)
                        writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)
                
            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            
            nn.utils.clip_grad_norm_(m.parameters(), 1.) # 梯度裁剪
            
            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0: # 每2000个step进行一次权重保存
                t.save({'model':m.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))

            
            


if __name__ == '__main__':
    main()