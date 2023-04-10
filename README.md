# ChatGLM-Uni
# Can Lora weights be used Exchange-Fusion in different large language models?
# Lora能在不同的大语言模型间交叉融合使用吗？(如何在小语言模型上得到接近大语言模型的效果研究)

## 介绍（Introduce）
近来，ChatGpt类语言模型凭借其优秀的人类友好性，以及令人诧异的通用性，引起业界的广泛关注。很多研究者也被其背后的原因所深深吸引。研究认为，此类模型的成功是三个因素的共同作用：1.模型参数量；2.指令格式数据集；3.增强学习方法。首先，模型参数量在GPT2以后被人们首先注意到，学者们发现模型参数量必须达到一定得体量，才能够超过“临界点”，或被称为“智能涌现”，获得愈发强大的能力，尤其是通用领域，而与此相反，若模型参数量过小，则学习很难记住，产生“灾难性遗忘”。目前来说，LLAMA-7B、ChatGLM-6B、ChatRWKV-7B的成功，侧面反应了以往GPT模型虽体积较大，但并未重复发挥出效用。第二，指令格式数据集是一种刚被创造不久的数据集，从认知学科的视角来看，其比以往数据集确实是非常独特的，甚至起到了中心性作用，关于这一点，我们会在后续的专题作出报告。第三，增强学习方法方面，此类大语言模型确从一种被称为“人类辅助强化学习”（RLHF）的新型学习算法上受益，其中以Lora微调最为常用，LLAMA模型上被使用次数也最多。

## 方法（Method）
This study proposes a Lora fine-tuning method suitable for ChatGPT class large language models. The main contribution is to attempt to use Lora weights as a starting point to strengthen the communication channels of different language models, in order to achieve complementary and complementary functions. We were mainly inspired by the Peft library and also benefited from its excellent fine-tuning and adaptation ability to different models. Taking ChatGLM-6B as an example, the official fine-tuning method is P-Tuning（ https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning ）Some researchers have also obtained affordable Lora fine-tuning methods based on the original trainer trainer of Transformer（ https://github.com/mymusise/ChatGLM-Tuning ）。 The Peft library supports all the fine-tuning methods mentioned above for ChatGLM. Our effort is to fine tune the LLAMA-7B model on the Chinese Alpaca dataset using Lora to obtain a Lora weight file, which is then embedded into the ChatGLM pre trained model. The verification shows that this is a simple and efficient technical solution that enhances the base of small models with large model weights. Similarly, regardless of whether the two models belong to the same class or not, it is also a considerable model interaction bridge. Overall, we have reported this method, known as the "Lora weight cross fusion", which not only has strong universality but also obvious effectiveness. More importantly, the cost of computing resources is very low.<br>
本研究提出了一种适用于ChatGPT类大语言模型的Lora微调方法，主要贡献是尝试以Lora权值为切入点，加强各种相异的语言大模型的沟通渠道，以此获得互补互足的功能。我们主要受到Peft库的启发，也得益于该库优秀对不同模型的微调适配能力。以ChatGLM-6B为例，官方的微调方法为P-Tuning,也有研究人员基于transformer原有的trainer训练器得到了平价的Lora微调方法。Peft库对ChatGLM支持以上所有微调方法。我们工作的努力，便是将LLAMA-7B模型在Chinese-Alpaca数据集上利用Lora进行微调，得到Lora权值文件，将该权值文件镶嵌进ChatGLM预训练模型中。验证表明，这是一种简介而高效的，以大模型权值增强小模型底座的技术方案，同理，无论两个模型是否属于同质或同级别模型，则又是一种可观的模型交互桥梁。总的来说，我们报告了这一方法，被称为“Lora权值交叉融合”，不但通用性较强，而且效果也显而易见，更重要的是，计算资源成本非常低廉。

## 实验环境(Trial Environment)
The main environment is as follows:<br>
Win10+Python310+Pytorch1.31+Cuda11.6<br>
Transformer (ChatGlm is using version 4.27.1, and pip3 is also installed. However, if you want to better use the LLAMA+Peft solution, it is recommended to use version 4.28, which can be downloaded from the official website or our warehouse)<br>
PEFT (0.2.0, already placed in this warehouse)<br>
Note: The pre training model of ChatGLM-6B has undergone changes compared to before on April 7, 2023, mainly in the VOCAB configuration and sub bin files of 01 and 08. It is recommended to use the new version of the pre training model. This warehouse stores a copy that supports Lora fusion.<br>

主要环境如下：<br>
Win10+Python310+Pytorch1.31+Cuda11.6<br>
Transformer(ChatGlm使用的是4.27.1版本，而pip3安装的也是，但如果想更好地使用LLAMA+Peft方案，建议使用4.28版本，可至官网，或本仓库内下载)<br>
PEFT(0.2.0，已放入本仓库)<br>
注意：ChatGLM-6B的预训练模型在2023.04.07较以往有变动，主要在vocab配置和01、08的子bin文件上，建议使用新版预训练模型。本仓库存储了一份支持Lora融合的复本。<br>

## 实验结果(Trial Result)
We found that ChatGlm-6B, mixed with Lora weight files, achieved the expected results in multiple text tasks such as comprehension, summarization, and continuation. The results show that using Lora weights from models with larger parameter values and incorporating the base of models with smaller parameter values can achieve more ideal results for models with smaller parameter values.<br>
我们发现经Lora权值文件混合的ChatGlm-6B，在理解、总结、续写等多个文本任务，均取得预期成效。结果表明，使用参数量较大模型的Lora权值，融入参数量较小模型的底座，能够使得参数量较小的模型效果得到更为理想的效果。<br>
![contact](resources/result.jpg)

## 讨论(Discussion)
We also noticed that LLAMA and ChatGLM use vastly different basic corpora (especially languages), and some people are concerned that this Alpaca lora approach may actually result in a decrease in the performance of ChatGLM in Chinese. In fact, this is not a problem caused by Lora weight cross fusion. Of course, we suggest that when choosing a new dataset, try to use the language you need (such as the Chinese Alpaca dataset), but this will not backfire. ChatRWKV testers report that the large model has amazing language structure learning and generalization ability, using only 1% Chinese corpus and 99% English corpus, and the model actually achieves "interoperability" in understanding Chinese and English. However, the reason why we recommend the language of the dataset is because the Lora weights will be mixed in Chinese and English after being assembled into the base, which is an easy problem to solve. You just need to tell it, 'Answer me in Chinese', and that's it.<br>
我们也注意到，LLAMA和ChatGLM由于所使用的基本语料（尤其是语言）大相径庭，有人担忧这种以Alpaca lora的方式，会反而使用ChatGLM中文性能下降。其实，这并不是Lora权值交叉融合造成的问题，当然我们建议，在选用新数据集时，尽量选用自己需要的语言（如中文Alpaca数据集），但这并不会让适得其反.ChatRWKV的测试者报告，大模型具有惊人的语言结构学习和泛化能力，仅使用1%的中文语料+99%的英文语料，模型反而对中英文理解达到“互通”。不过，我们之所以建议数据集的语种，是因为Lora权值在组装进底座后，会存在“中英文混用”，这个问题很容易解决，你只要告诉它，“用中文回答我”，就可以了。

## 结论与后续课题(Conclusion and Future Work)
This study preliminarily attempts to use the peft fine-tuning library to construct interaction bridges for ChatGpt class large language models of different types, volumes, and configurations through the Lora weight fusion method. Our experiment selected ChatGlm-6B as the base small model, while the slightly larger LLAMA-7B was translated into Chinese into the Alpaca dataset, and the Lora weights obtained by the Lora method were used as components. Finally, the Lora weights were integrated into ChatGlm-6B, and preliminary results showed that this technical approach achieved satisfactory results. Subsequently, we also found that the Lora weight cross fusion strategy has excellent transfer ability, and its robustness is also excellent due to its reliance on a powerful Peft library.In the future, we will continue to obtain Lora weights on different datasets, not limited to the paradigm of "LLAMA Lora+ChatGLM pre training model". Meanwhile, as mentioned earlier, we will further publish research on instruction datasets and, if possible, add brain like computing components, hoping to promote the "intelligent emergence" of small language models (SLM).<br>
本研究初步尝试利用peft微调库，通过Lora权值融合的方法，构造起不同种类、不同体积、不同配置的ChatGpt类大语言模型的交互桥梁。我们的实验选取了ChatGlm-6B为底座小模型，而以稍大些的LLAMA-7B经中文翻译Alpaca数据集，使用Lora方法得到的Lora权值作为组件，最后将该Lora权值整合进ChatGlm-6B，初步来看，该技术手段取得了令人满意的功效。随后，我们还发现，Lora权值交叉融合策略，具有极好的迁移能力，再者因依托于强大的Peft库，因而鲁棒性也比较优异。后续，我们将继续在不同的数据集上获得Lora权值，并不仅仅限于“LLAMA Lora+ChatGLM预训练模型”这种范式。与此同步地，正如前文所言，我们会进一步公布有关指令数据集的研究，如有条件则会加入类脑计算组件，希冀推动小语言模型（SLM）的“智能涌现”进展。

## 更新日志(Update Log)
2023-04-10<br>
We were pleasantly surprised to find that the Lora fusion method has a more widespread application range than we had anticipated. Therefore, in the published web version, LLAMA-13B trained by Alpaca has been grafted and assembled into ChatGLM-6B.<br>
我们惊喜地发现，Lora融合法的适用范围比我们预想的更为普遍，因而在发布的网页版中，已将由Alpaca训练的LLAMA-13B，嫁接组装进ChatGLM-6B中。<br><br>

2023-04-09<br>
The initial start of the project can use the weights trained by LLAMA Alpaca for ChatGlm. Compared to LLAMA, ChatGlm has a smaller model size and deployment cost, which is close to the effect of LLAMA, especially in terms of English performance.<br>
工程初步启动，可使用LLAMA Alpaca训练的权值，用于ChatGlm，相比LLAMA，ChatGlm以更小的模型体积和部署成本，接近于LLAMA的效果，尤其在英文的表现上。<br><br>

## 共创共赢(Join This Work)
If you are interested in our work, please give "Fork" or "Star" your attention and support. We will be grateful much, or contact us. QR code:<br>
如您对我们的工作产生兴趣，请给予小星星Star关注和支持，我们将不尽感激，或与我们联系，QQ群二维码：<br>
![contact](resources/QQgroup.jpg)


## 致谢(Acknowledgments)
[1]ChatRWKV:https://github.com/BlinkDL/ChatRWKV<br>
[2]ChatGLM:https://github.com/THUDM/ChatGLM-6B<br>
[3]Alpaca:https://github.com/tatsu-lab/stanford_alpaca<br>
