import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from gpt import GPT
import pandas as pd
import os

# 定义情感分类数据集类
class EmotionDataset(Dataset):
    def __init__(self, csv_file, vocab_path, max_length=512):
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'review']
        label = self.data.loc[idx, 'label']
        # 将文本转换为索引
        tokens = [self.vocab.get(char, self.vocab['<unk>']) for char in text]
        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.vocab['<pad>']] * (self.max_length - len(tokens))
        input_ids = torch.tensor(tokens)
        label = torch.tensor(label, dtype=torch.long)
        return input_ids, label

def freeze_parameters(model, freeze_type='all'):
    """
    冻结模型参数
    freeze_type:
        'all' - 训练所有层
        'last' - 仅训练输出层和最后一个 Transformer 块
    """
    if freeze_type == 'all':
        for param in model.parameters():
            param.requires_grad = True
    elif freeze_type == 'last':
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后一个 Transformer 块
        for name, param in model.named_parameters():
            if 'layers.5' in name:  # 假设有6个层，索引从0开始
                param.requires_grad = True
        # 解冻输出层参数
        for param in model.out.parameters():
            param.requires_grad = True
    else:
        raise ValueError("freeze_type 必须是 'all' 或 'last'")

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10, save_dir='emotion_checkpoints', start_epoch=0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_checkpoints = []

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            # 获取序列的第一个位置的输出（或使用池化）
            logits = outputs[:, 0, :]  # [batch_size, vocab_size]
            # 添加分类层
            pred = model.classifier(logits)  # [batch_size, 2]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 计算准确率
            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx} / {len(train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # 保存检查点
        checkpoint_name = f'epoch_{epoch+1}-loss_{avg_loss:.4f}-acc_{accuracy:.2f}.pt'
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)

        saved_checkpoints.append(checkpoint_path)

        # 如果保存的检查点超过五个，删除最早的一个
        if len(saved_checkpoints) > 5:
            oldest_checkpoint = saved_checkpoints.pop(0)
            os.remove(oldest_checkpoint)

def main():
    # 超参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 10
    FREEZE_TYPE = 'last'  # 可选 'all', 'last'

    # 加载数据集
    train_dataset = EmotionDataset('./data/emotion_train.csv', './data/emotion_vocab.json')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 加载词汇表
    with open('./data/emotion_vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载预训练的GPT模型
    model = GPT(len(vocab)).to(device)
    checkpoint = torch.load('checkpoints/best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 添加分类层
    model.classifier = nn.Linear(model.out.out_features, 2).to(device)

    # 冻结参数
    freeze_parameters(model, freeze_type=FREEZE_TYPE)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    train_model(model, train_loader, optimizer, criterion, device, NUM_EPOCHS)

if __name__ == '__main__':
    main()