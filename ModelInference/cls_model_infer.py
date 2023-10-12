"""
@Author: pkq1688
@Date: 2023-08-20 17:07:10
@LastEditors: pkq1688
@LastEditTime: 2023-08-20 17:10:23
@FilePath: /RewriteLLM/ModelInference/cls_model_infer.py
@Description: 
"""
from typing import List

import torch
from rich import print
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def inference(
    model, tokenizer, sentences: List[str], device: str, batch_size=16, max_seq_len=128
) -> List[int]:
    """
    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        sentences (List[str]): _description_
        batch_size (int, optional): _description_. Defaults to 16.
        max_seq_len (int, optional): _description_. Defaults to 128.

    Returns:
        List[int]: [label1, label2, label3, ...]
    """
    res = list()
    for i in range(0, len(sentences), batch_size):
        batch_sentence = sentences[i : i + batch_size]
        ipnuts = tokenizer(
            batch_sentence,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        output = model(
            input_ids=ipnuts["input_ids"].to(device),
            token_type_ids=ipnuts["token_type_ids"].to(device),
            attention_mask=ipnuts["attention_mask"].to(device),
        ).logits
        # output = torch.argmax(output, dim=-1).cpu().tolist()
        output = torch.sigmoid(output).cpu().tolist()
        # print(output)
        output = [one[1] for one in output]
        res.extend(output)
    return res


if __name__ == "__main__":
    device = "cuda:0"  # 指定GPU设备
    saved_model_path = "checkpoints/wyw_classify/model_best"  # 训练模型存放地址
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)
    model.to(device).eval()

    sentences = [
        "六王毕，四海一，蜀山兀，阿房出",
        "秦人不暇自哀，而后人哀之。后人哀之而不鉴之，亦使后人而复哀后人也。",
        "任何一部分肌肤，任何一种姿容，都娇媚极了。",
    ]
    res = inference(model, tokenizer, sentences, device)
    print("res: ", res)
