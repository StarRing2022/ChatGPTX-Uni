# ChatGLM-Uni
# Can Lora weights be used Exchange-Fusion in different large language models?
# Lora能在不同的大语言模型间交叉融合使用吗？

## 介绍（Introduce）
近来，ChatGpt类语言模型凭借其优秀的人类友好性，以及令人诧异的通用性，引起业界的广泛关注。很多研究者也被其背后的原因所深深吸引。研究认为，此类模型的成功是三个因素的共同作用：1.模型参数量；2.指令格式数据集；3.增强学习方法。首先，模型参数量在GPT2以后被人们首先注意到，学者们发现模型参数量必须达到一定得体量，才能够超过“临界点”，或被称为“智能涌现”，获得愈发强大的能力，尤其是通用领域，而与此相反，若模型参数量过小，则学习很难记住，产生“灾难性遗忘”。目前来说，LLAMA-7B、ChatGLM-6B、ChatRWKV-7B的成功，侧面反应了以往GPT模型虽体积较大，但并未重复发挥出效用。第二，指令格式数据集是一种刚被创造不久的数据集，从认知学科的视角来看，其比以往数据集确实是非常独特的，甚至起到了中心性作用，关于这一点，我们会在后续的专题作出报告。第三，增强学习方法方面，此类大语言模型确从一种被称为“人类辅助强化学习”（RLHF）的新型学习算法上受益，其中以Lora微调最为常用，在LLAMA模型上被使用的次数也最多。

## 方法（Method）

## 实验环境(Trial Environment)

## 实验结果(Trial Result)
我们发现经Lora权值文件混合的ChatGlm-6B，在理解、总结、续写等多个文本任务，均取得预期成效。结果表明，使用参数量较大模型的Lora权值，融入参数量较小模型的底座，能够使得参数量较小的模型效果得到更为理想的效果。
![contact](resources/result.jpg)

## 讨论(Trial Result)
我们也注意到，LLAMA和ChatGLM由于所使用的基本语料（尤其是语言）大相径庭，有人担忧这种以Alpaca lora的方式，会反而使用ChatGLM中文性能下降。其实，这并不是Lora权值交叉融合造成的问题，当然我们建议，在选用新数据集时，尽量选用自己需要的语言（如中文Alpaca数据集），但这并不会让适得其反.ChatRWKV的测试者报告，大模型具有惊人的语言结构学习和泛化能力，仅使用1%的中文语料+99%的英文语料，模型反而对中英文理解达到“互通”。不过，我们之所以建议数据集的语种，是因为Lora权值在组装进底座后，会存在“中英文混用”，这个问题很容易解决，你只要告诉它，“用中文回答我”，就可以了。

## 结论与后续课题(Conclusion and Future Work)
本研究初步尝试利用peft微调库，通过Lora权值融合的方法，构造起不同种类、不同体积、不同配置的ChatGpt类大语言模型的交互桥梁。我们的实验选取了ChatGlm-6B为底座小模型，而以稍大些的LLAMA-7B经中文翻译Alpaca数据集，使用Lora方法得到的Lora权值作为组件，最后将该Lora权值整合进ChatGlm-6B，初步来看，该技术手段取得了令人满意的功效。随后，我们还发现，Lora权值交叉融合策略，具有极好的迁移能力，再者因依托于强大的Peft库，因而鲁棒性也比较优异。

## 更新日志(Update Log)
2023-04-10<br>

我们惊喜地发现，Lora融合法的适用范围比我们预想的更为普遍，因而在发布的网页版中，已将由Alpaca训练的LLAMA-13B，嫁接组装进ChatGLM-6B中.<br><br>

2023-04-09<br>
The initial start of the project can use the weights trained by LLAMA Alpaca for ChatGlm. Compared to LLAMA, ChatGlm has a smaller model size and deployment cost, which is close to the effect of LLAMA, especially in terms of English performance.<br>
工程初步启动，可使用LLAMA Alpaca训练的权值，用于ChatGlm，相比LLAMA，ChatGlm以更小的模型体积和部署成本，接近于LLAMA的效果，尤其在英文的表现上.<br><br>

## 共创共赢(Join This Work)
If you are interested in our work, please give "Fork" or "Star" your attention and support. We will be grateful much, or contact us. QR code:<br>
如您对我们的工作产生兴趣，请给予小星星Star关注和支持，我们将不尽感激，或与我们联系，QQ群二维码：<br>
![contact](resources/QQgroup.jpg)


## 致谢(Acknowledgments)
[1]ChatRWKV:https://github.com/BlinkDL/ChatRWKV<br>
[2]ChatGLM:https://github.com/THUDM/ChatGLM-6B<br>
[3]Alpaca:https://github.com/tatsu-lab/stanford_alpaca<br>
