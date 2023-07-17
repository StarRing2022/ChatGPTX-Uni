import argparse
import json
from tqdm import tqdm

import datasets
import transformers

#将jsonl转换为数据集文件夹


def preprocess(tokenizer, config, example, max_seq_length):
    # print(example["context"])
    # print(example["target"])
    prompt = example["context"].split("Answer")[0]
    #print(prompt)
    target = example["context"].split("Answer")[1]
    #print(target)
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=False):
    model_name = "chatglm-6b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            feature["seq_len"] =  feature["seq_len"]
            yield feature
    return feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/alpaca_data.jsonl")
    parser.add_argument("--save_path", type=str, default="data/alpaca_data")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(args.jsonl_path, args.max_seq_length, args.skip_overlength)
    )
    #print(dataset)
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
