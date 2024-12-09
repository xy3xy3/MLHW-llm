import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from emotion_test import EmotionTestDataset, evaluate_model
from gpt import GPT
import pandas as pd
import os

class EmotionDataset(Dataset):
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

def freeze_parameters(model, freeze_type='all'):
    if freeze_type == 'all':
        for param in model.parameters():
            param.requires_grad = True
    elif freeze_type == 'last':
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'layers.5' in name:
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError("freeze_type 必须是 'all' 或 'last'")

def train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=10, save_dir='emotion_checkpoints', start_epoch=0, freeze_type='all'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saved_checkpoints = []
    test_accuracies = []  # Initialize test_accuracies
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith(f'-{freeze_type}.pt')]
    if checkpoint_files and start_epoch == 0:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('-')[0]))
        checkpoint_path = os.path.join(save_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming from epoch {start_epoch}')
    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            pooled_output = torch.mean(outputs, dim=1)
            pred = model.classifier(pooled_output)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = (torch.sigmoid(pred) > 0.5).float()
            total += y.size(0)
            # print(f"预测 {predicted.squeeze()} 实际 {y}")
            correct += (predicted.squeeze() == y).sum().item()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx} / {len(train_loader)}, Loss: {loss.item():.4f}')
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        msg = f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%\n' + f'Epoch {epoch} Test Accuracy: {test_accuracy:.2f}%'
        print(msg)
        with open(os.path.join(save_dir, f'emft-{freeze_type}.txt'), 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        checkpoint_name = f'epoch_{epoch+1}-loss_{avg_loss:.4f}-train_acc_{accuracy:.2f}-test_acc_{test_accuracy:.2f}-{freeze_type}.pt'
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'test_accuracy': test_accuracy,
        }, checkpoint_path)
        saved_checkpoints.append(checkpoint_path)
        if len(saved_checkpoints) > 5:
            oldest_checkpoint = saved_checkpoints.pop(0)
            os.remove(oldest_checkpoint)
    with open(os.path.join(save_dir, 'test_accuracies.txt'), 'w') as f:
        for acc in test_accuracies:
            f.write(f'{acc}\n')

def load_pretrained_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    pretrained_embed = pretrained_dict['embedding.weight']
    current_embed = model_dict['embedding.weight']
    current_embed[:pretrained_embed.size(0)] = pretrained_embed
    pretrained_out_w = pretrained_dict['out.weight']
    pretrained_out_b = pretrained_dict['out.bias']
    current_out_w = model_dict['out.weight']
    current_out_b = model_dict['out.bias']
    current_out_w[:pretrained_out_w.size(0)] = pretrained_out_w
    current_out_b[:pretrained_out_b.size(0)] = pretrained_out_b
    for name, param in pretrained_dict.items():
        if name not in ['embedding.weight', 'out.weight', 'out.bias']:
            model_dict[name].copy_(param)
    model.load_state_dict(model_dict)
    return model

def main():
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 20
    FREEZE_TYPE = 'all'
    train_dataset = EmotionDataset('./data/emotion_train.csv', './data/emotion_vocab.json')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = EmotionTestDataset('./data/emotion_test.csv', './data/emotion_vocab.json')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    with open('./data/emotion_vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(len(vocab)).to(device)
    model = load_pretrained_model(model, 'checkpoints/best.pt', device)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.out.out_features, 1)
    ).to(device)
    freeze_parameters(model, freeze_type=FREEZE_TYPE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    train_model(model, train_loader, test_loader, optimizer, criterion, device, NUM_EPOCHS, freeze_type=FREEZE_TYPE)

if __name__ == '__main__':
    main()