# ChatGPTX-Uni

## Lora能在不同的大语言模型间集成切换使用吗？(LLAMA-Lora & GLM-Lora集成增强学习)

1.工程介绍：<br>
为找到一种在每种GPT类LLM大语言模型的通用交流方案，并能使得不同模型之间进行互补互足，发挥出集成优势。实现一种多Lora权值集成切换+Zero-Finetune零微调增强的跨模型技术方案，LLM-Base+LLM-X+Alpaca，初期，LLM-Base为Chatglm6B底座模型，LLM-X是LLAMA增强模型。理论上，任何支持HF格式，可以使用Peft库的两个甚至三个以上的LLM模型都可以利用此方法进行交流和集成。<br>
注意：Peft库新版有变动，请使用本工程的RingPeft库，其基于Peft0.2，但又添入了新版Peft的一些Lora方法。<br>

2.主要代码说明：<br>
llama-finetune.py：基于原英文Alpaca数据集，LLAMA 7B的Lora微调
glm-finetunejsonl.py：基于test.jsonl数据集，GLM 6B的Lora微调
cover_alpaca2jsonl.py：将json数据集转为jsonl数据集
tokenize_dataset_rowsjsonl.py：对jsonl数据集转为transfomers的datasets文件夹
zerofinetune.py：在不进行训练的情况下，仅使用提示工程进行微调
generate.py：架设网页服务

3.使用：<br>
环境：WIN10+Torch1.31+Cuda11.6<br>
python generate.py<br>
test.json仅为2条数据，训练100个epoch，仅供测试
