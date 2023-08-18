"""
@Author       : pkq1688
@email        : adolf1321794021@gmail.com
@Date         : 2023-04-20 14:23:34
@LastEditors: pkq1688
@LastEditTime: 2023-08-18 15:05:04
@Description  : 
"""
from pathlib import Path

def handle_base_data():
    file_data_path = Path("dataset/Classical-Modern/双语数据")

    gw_list = []
    bh_list = []

    for file_path in file_data_path.rglob("bitext.txt"):
        print(f"Found target file: {file_path}")
        # with open()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "古文：" in line:
                    gw_list.append(line.replace("古文：", "").strip())
                if "现代文：" in line:
                    bh_list.append(line.replace("现代文：", "").strip())
            
            assert len(gw_list) == len(bh_list)

    with open("dataset/classical_cls/cls_data.txt","w") as f:
        for gw, bh in zip(gw_list, bh_list):
            f.write(f"{gw}\t1\n")
            f.write(f"{bh}\t0\n")