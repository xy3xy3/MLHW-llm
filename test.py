import json
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 参数设置
MAX_SEQ_LENGTH = 128
CHECKPOINT_DIR = './checkpoints'

# 加载词汇表
with open('./data/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# 反向词汇表
idx2char = {int(idx): char for idx, char in json.load(open('./data/idx2char.json', 'r', encoding='utf-8')).items()}

# 特殊标记ID
PAD_ID = vocab.get("<pad>")
UNK_ID = vocab.get("<unk>")
SEP_ID = vocab.get("<sep>")

# 配置GPT模型
config = GPT2Config(
    vocab_size=len(vocab),
    n_positions=MAX_SEQ_LENGTH,
    n_ctx=MAX_SEQ_LENGTH,
    n_embd=256,           # 嵌入维度
    num_hidden_layers=2,  # 解码器层数
    num_attention_heads=4 # 注意力头数
)

# 加载模型
model = GPT2LMHeadModel(config)
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "final_model.pt"), map_location=device))
model.to(device)
model.eval()

# 编码函数
def encode(text):
    return [vocab.get(char, UNK_ID) for char in text]

# 解码函数
def decode(ids):
    return ''.join([idx2char[idx] for idx in ids])

# 生成文本函数
def generate_text(prompt, max_length=50):
    input_ids = encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 生成文本
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = decode(output[0].tolist())

    return generated_text

# 测试生成文本
prompt = input("请输入文本")
generated_text = generate_text(prompt, max_length=50)
print(f"输入: {prompt}")
print(f"生成: {generated_text}")