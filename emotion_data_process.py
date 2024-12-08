import pandas as pd
import json
from sklearn.utils import shuffle

# 读取原始预训练模型的词汇表
with open('./data/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# 读取数据
data = pd.read_csv('./data/ChnSentiCorp_htl_all.csv')

# 打乱数据
data = shuffle(data, random_state=42).reset_index(drop=True)

# 分别获取标签为0和1的数据
data_0 = data[data['label'] == 0].reset_index(drop=True)
data_1 = data[data['label'] == 1].reset_index(drop=True)

# 保留最后250条0和1的数据作为测试集
test_data_0 = data_0.tail(250)
test_data_1 = data_1.tail(250)
test_data = pd.concat([test_data_0, test_data_1], ignore_index=True)

# 将测试集保存为CSV文件
test_data.to_csv('./data/emotion_test.csv', index=False)

# 剩余的数据作为训练集，各取3000条0和1的数据
train_data_0 = data_0.head(3000)
train_data_1 = data_1.head(3000)
train_data = pd.concat([train_data_0, train_data_1], ignore_index=True)

# 保存训练集
train_data.to_csv('./data/emotion_train.csv', index=False)

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