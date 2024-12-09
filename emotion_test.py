import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from gpt import GPT

# 定义测试数据集类
class EmotionTestDataset(Dataset):
    def __init__(self, csv_file, vocab_path, max_length=512):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.data = pd.read_csv(csv_file, dtype={'review': str, 'label': int})
        self.data['review'] = self.data['review'].fillna('')
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'review']
        label = self.data.loc[idx, 'label']
        if not isinstance(text, str):
            text = str(text)
        tokens = [self.vocab.get(char, self.vocab['<unk>']) for char in text]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.vocab['<pad>']] * (self.max_length - len(tokens))
        input_ids = torch.tensor(tokens)
        label = torch.tensor(label, dtype=torch.float)
        return input_ids, label

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pooled_output = torch.mean(outputs, dim=1)
            pred = model.classifier(pooled_output)
            predicted = (torch.sigmoid(pred) > 0.5).float()
            total += y.size(0)
            correct += (predicted.squeeze() == y).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

def main():
    BATCH_SIZE = 32

    # 加载测试数据集
    test_dataset = EmotionTestDataset('./data/emotion_test.csv', './data/emotion_vocab.json')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载词汇表
    with open('./data/emotion_vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载微调后的模型
    model = GPT(len(vocab)).to(device)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.out.out_features, 1)
    ).to(device)
    checkpoint = torch.load('emotion_checkpoints/last.pt', map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 评估模型
    res = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {res:.2f}%")

if __name__ == '__main__':
    main()