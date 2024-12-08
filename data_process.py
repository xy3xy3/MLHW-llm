import json

# 第1步：读取数据
data_file = './data/train.txt'
with open(data_file, 'r', encoding='utf-8') as f:
    raw_data = f.read()

# 分割句子，移除空白句子
sentences = [line.strip() for line in raw_data.split('\n') if line.strip()]

# 第2步：创建词汇表
# 提取所有唯一字符
chars = set(''.join(sentences))

# 定义特殊标记
special_tokens = ['<pad>', '<unk>', '<sep>']

# 创建字符到ID的映射
vocab = {char: idx for idx, char in enumerate(special_tokens)}
start_idx = len(vocab)
for idx, char in enumerate(chars, start=start_idx):
    vocab[char] = idx

# 第3步：生成反向映射
id2char = {idx: char for char, idx in vocab.items()}

# 第4步：保存词汇表和映射
with open('./data/vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False)

with open('./data/id2char.json', 'w', encoding='utf-8') as f:
    json.dump(id2char, f, ensure_ascii=False)

print("词汇表和映射已保存。")