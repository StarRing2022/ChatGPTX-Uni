from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "who are you?", history=[])
print(response)

# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)