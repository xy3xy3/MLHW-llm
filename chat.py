import torch
import torch.nn as nn
import json
from gpt import GPT
import torch.nn.functional as F

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    idx2char = {idx: char for char, idx in vocab.items()}
    return vocab, idx2char

def generate_reply(model, vocab, idx2char, input_text, device, max_length=50):
    model.eval()
    tokens = [vocab.get(char, vocab['<unk>']) for char in input_text]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            if next_token_id.item() == vocab['<sep>']:
                break

    generated_tokens = input_ids[0].tolist()
    generated_text = ''.join([idx2char.get(token_id, '') for token_id in generated_tokens])
    reply = generated_text[len(input_text):]  # 去除输入部分，保留模型生成的回复
    return reply

def main():
    # 加载词汇表
    vocab, idx2char = load_vocab('./data/vocab.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = GPT(len(vocab)).to(device)
    checkpoint = torch.load('checkpoints/best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("欢迎使用终端聊天机器人，输入'退出'结束对话。")
    while True:
        user_input = input("用户：")
        if user_input == '退出':
            print("聊天已结束。")
            break
        reply = generate_reply(model, vocab, idx2char, user_input, device)
        print(f"机器人：{reply}")

if __name__ == '__main__':
    main()