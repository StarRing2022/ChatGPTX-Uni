from transformers import AutoTokenizer, AutoModel
import json


# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

tokenizer = AutoTokenizer.from_pretrained("chatglm-6b", trust_remote_code=False)
model = AutoModel.from_pretrained("chatglm-6b", trust_remote_code=False).half().cuda()

responseold, history = model.chat(tokenizer, "给出三个保持健康的小贴士。", history=[])
print(responseold)

#使用"我"的效果要比"你"更好
responselearn, history = model.chat(tokenizer,"我是一名学习者，现在我需要学习和记忆以下内容：" + "给出三个保持健康的小贴士。", history=[["给出三个保持健康的小贴士。","1. 饮食要均衡，确保摄入足够的水果和蔬菜。\n2. 经常锻炼，保持身体活跃和强壮。\n3. 要保证充足的睡眠，并保持一个稳定的睡眠时间表。"]])
print(responselearn)

responsenew, history = model.chat(tokenizer,"利用我已经知道的，和刚刚学会的内容，回答："+"给出三个保持健康的小贴士。", history=[])
print(responsenew)


#将prompt和responsenew写入新的数据集

data = [
        {
        'instruction': "给出三个保持健康的小贴士。",
        'input': "",
        'output': responsenew
        }
    ]
   
with open('./data/zerofinetune-demo.json', 'w',encoding='utf-8') as f:
    json.dump(data,f,ensure_ascii=False)


