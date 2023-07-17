from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel,DataCollatorForLanguageModeling,DataCollatorForSeq2Seq
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os

#使用jsonl作为数据集

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


tokenizer = AutoTokenizer.from_pretrained("chatglm-6b", trust_remote_code=True,device_map='auto')


@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca_data")
    model_path: str = field(default="output")
    lora_rank: int = field(default=16) #这里原本是8，要改为16


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features: list) -> dict:
    
    len_ids = [len(feature["input_ids"]) for feature in features]
    
    longest = max(len_ids)
    input_ids = []
    labels_list = []

    #print(features)
    
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():
    writer = SummaryWriter()
    finetune_args = FinetuneArguments
    

    # init model
    model = AutoModel.from_pretrained(
        "chatglm-6b", load_in_8bit=False, trust_remote_code=True, device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # setup peft
    

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['query_key_value'] 
    )
    model = get_peft_model(model, peft_config)

    model = model.cuda()

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    #print(dataset.features)
    #print(f"\n{len(dataset)=}\n")

    # start train
    training_args = TrainingArguments(
        output_dir="lora-out",
        fp16 =True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size = 1,
        learning_rate = 1e-4,
        #num_train_epochs = 1,
        #max_steps=1500,
        max_steps=100,
        logging_steps=50,
        remove_unused_columns=False,
        seed=0,
        data_seed=0,
        group_by_length=False,
)
    
    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[TensorBoardCallback(writer)],

        #法1：GLM微调方法
        data_collator=data_collator,

        #法2：GPT微调方法
        #data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
