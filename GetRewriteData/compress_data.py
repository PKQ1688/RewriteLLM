"""
@Author: pkq1688
@Date: 2023-08-20 23:57:14
@LastEditors: pkq1688
@LastEditTime: 2023-08-21 21:55:53
@FilePath: /RewriteLLM/GetRewriteData/compress_data.py
@Description: 
"""
import os

import json
import zstandard as zstd


SHARD_SIZE = 10


def compress(input_file: str, write_path: str):
    """
    将数据压缩成.zst格式的文件，以支持流氏读取。
    """
    print(f"processed {input_file}...")

    path_name = os.path.dirname(write_path)
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    total_num, file_num, log_interval = 0, 1, 10000
    wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
    with open(input_file, "r") as f:
        for line in f:
            line = json.loads(line)
            if total_num % SHARD_SIZE == 0 and total_num > 0:
                file_num += 1
                wfp.close()
                wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
            wfp.write(json.dumps(line).encode("utf-8"))
            wfp.write("\n".encode("utf-8"))
            total_num += 1
            if not total_num % log_interval:
                print(f"\rProcessed: {total_num} samples...", end="")
    wfp.close()
    print("total line: {}\ntotal files: {}".format(total_num, file_num))


def batch_compress_pretrain_data():
    """
    批量压缩预训练数据。
    """
    source_path = "dataset/Classical-Modern/古文原文"  # 源数据文件
    target_path = "dataset/pretrain_data"  # 压缩后存放地址

    files = ["MNBVC_news", "MNBVC_qa", "MNBVC_wiki"]

    compress_file = []
    for file in files:
        compress_file.append(
            {
                "input_file": f"{source_path}/{file}.jsonl",
                "write_path": f"{target_path}/{file}/part-{{}}.jsonl.zst",
            }
        )

    for file in compress_file:
        compress(file["input_file"], file["write_path"])


def batch_compress_sft_data():
    """
    批量压缩SFT数据。
    """
    source_path = "shuffled_data/sft"
    target_path = "sft_data"

    files = ["sharegpt"]

    compress_file = []
    for file in files:
        compress_file.append(
            {
                "input_file": f"{source_path}/{file}.jsonl",
                "write_path": f"{target_path}/{file}/part-{{}}.jsonl.zst",
            }
        )

    for file in compress_file:
        compress(file["input_file"], file["write_path"])


if __name__ == "__main__":
    batch_compress_pretrain_data()
    # batch_compress_sft_data()
