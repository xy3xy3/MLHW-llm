import os
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
import math

# 数据集类
class TextDataset(Dataset):
    def __init__(self, file_path, vocab_path, max_length=512):
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # 加载数据文件
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read().split('\n')
            # 过滤掉长度超过64的句子
            self.data = [x for x in self.data if x.strip() and len(x.strip()) <= 64]
            # 过滤掉长度小于16的句子
            self.data = [x for x in self.data if len(x.strip()) >= 16]

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # 将文本转换为词汇表中的索引
        tokens = [self.vocab.get(char, self.vocab['<unk>']) for char in text]

        # 如果句子长度超过最大长度，截断
        if len(tokens) > self.max_length - 1:
            tokens = tokens[:self.max_length-1]

        # 填充到最大长度
        tokens = tokens + [self.vocab['<pad>']] * (self.max_length - len(tokens))

        # 输入序列和目标序列
        x = torch.tensor(tokens[:-1])
        y = torch.tensor(tokens[1:])

        return x, y

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入张量上
        return x + self.pe[:, :x.size(1)]

# GPT解码器层
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力机制
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# GPT模型
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1):
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        # 解码器层
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出层
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 生成因果注意力掩码
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)

        # 词嵌入和位置编码
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 转换为[seq_len, batch, d_model]格式
        x = x.transpose(0, 1)

        # 通过解码器层
        for layer in self.layers:
            x = layer(x, mask)

        # 转换回[batch, seq_len, d_model]格式
        x = x.transpose(0, 1)

        # 输出层
        return self.out(x)

# 训练代码
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10, save_dir='checkpoints', start_epoch=0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_checkpoints = []  # 用于存储已保存的检查点文件路径

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx} / {len(train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')

        # 保存检查点，文件名包含epoch编号和损失值
        checkpoint_name = f'epoch_{epoch+1}-loss_{avg_loss:.4f}.pt'
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)

        saved_checkpoints.append(checkpoint_path)

        # 如果保存的检查点超过五个，删除最早的那一个
        if len(saved_checkpoints) > 5:
            oldest_checkpoint = saved_checkpoints.pop(0)
            os.remove(oldest_checkpoint)

# 主函数
def main():
    # 超参数
    BATCH_SIZE = 96  # 每个批次的样本数量
    D_MODEL = 256  # Embedding层和解码器层的维度
    N_HEADS = 8  # 多头注意力机制的头数
    N_LAYERS = 6  # 解码器层的数量
    D_FF = 1024  # 前馈神经网络的隐藏层维度
    DROPOUT = 0.1  # Dropout的概率
    LEARNING_RATE = 0.0001  # 学习率
    NUM_EPOCHS = 50  # 训练的轮数

    # 加载数据
    dataset = TextDataset('./data/train.txt', './data/vocab.json')
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    with open('./data/vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(len(vocab), D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    # 检查是否有之前的检查点文件
    save_dir = 'checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pt') and f.startswith('epoch_')]
    if checkpoint_files:
        # 找到最新的检查点文件
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('-')[0]))
        checkpoint_path = os.path.join(save_dir, latest_checkpoint)

        # 加载检查点
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming training from epoch {start_epoch}')
    else:
        start_epoch = 0

    # 训练模型
    train_model(model, train_loader, optimizer, criterion, device, NUM_EPOCHS, save_dir, start_epoch)

if __name__ == '__main__':
    main()