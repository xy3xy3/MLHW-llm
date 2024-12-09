import pandas as pd
import json
from sklearn.utils import shuffle

# 读取原始预训练模型的词汇表
with open('./data/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# 读取数据
data = pd.read_csv('./data/ChnSentiCorp_htl_all.csv')

# 去掉包含缺失值的样本
data = data.dropna(subset=['review', 'label']).reset_index(drop=True)

# 打乱数据
data = shuffle(data, random_state=42).reset_index(drop=True)

# 分别获取标签为0和1的数据
data_0 = data[data['label'] == 0].reset_index(drop=True)
data_1 = data[data['label'] == 1].reset_index(drop=True)

# 取数量较少的一类的数量
n_samples = min(len(data_0), len(data_1))

# 从两类中各取n_samples条数据
balanced_data_0 = data_0.head(n_samples)
balanced_data_1 = data_1.head(n_samples)

# 合并为平衡数据集
balanced_data = pd.concat([balanced_data_0, balanced_data_1], ignore_index=True)

# 从平衡数据集中随机选择250个正样本和250个负样本作为测试集
test_data_0 = balanced_data_0.sample(n=250, random_state=42)
test_data_1 = balanced_data_1.sample(n=250, random_state=42)
test_data = pd.concat([test_data_0, test_data_1], ignore_index=True)

# 剩余的数据作为训练集
train_data_0 = balanced_data_0.drop(test_data_0.index).reset_index(drop=True)
train_data_1 = balanced_data_1.drop(test_data_1.index).reset_index(drop=True)
train_data = pd.concat([train_data_0, train_data_1], ignore_index=True)

print(f"每类训练样本数量: {len(train_data_0)}")
print(f"每类测试样本数量: {len(test_data_0)}")

# 保存训练集和测试集
train_data.to_csv('./data/emotion_train.csv', index=False)
test_data.to_csv('./data/emotion_test.csv', index=False)

# 扩展词汇表：将情感数据集中的新字符添加到现有词汇表中
all_text = ''.join(map(str, train_data['review']))
chars = set(all_text)

# 添加新字符到词汇表
max_id = max(vocab.values())
for char in chars:
    if char not in vocab:
        max_id += 1
        vocab[char] = max_id

# 生成反向映射
id2char = {idx: char for char, idx in vocab.items()}

# 保存扩展后的词汇表和映射
with open('./data/emotion_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False)

with open('./data/emotion_id2char.json', 'w', encoding='utf-8') as f:
    json.dump(id2char, f, ensure_ascii=False)

print("情感数据的词汇表和映射已保存。")
print(f"词汇表大小: {len(vocab)}")