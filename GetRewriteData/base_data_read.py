"""
@Author: pkq1688
@Date: 2023-08-18 17:13:10
@LastEditors: pkq1688
@LastEditTime: 2023-08-18 17:14:30
@FilePath: /RewriteLLM/GetRewriteData/base_data_read.py
@Description: 
"""
import numpy as np
import traceback


def convert_example(examples: dict, tokenizer, max_seq_len: int):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '1	这是一条正样本',
                                                            '0	这是一条负样本',
                                                            ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]],
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'labels': [1, 0, ...]
                        }
    """
    tokenized_output = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for example in examples["text"]:
        try:
            label, content = example.split("\t")
            encoded_inputs = tokenizer(
                text=content,
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
            )
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

        tokenized_output["input_ids"].append(encoded_inputs["input_ids"])
        tokenized_output["token_type_ids"].append(encoded_inputs["token_type_ids"])
        tokenized_output["attention_mask"].append(encoded_inputs["attention_mask"])
        tokenized_output["labels"].append(int(label))

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output
