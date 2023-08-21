"""
@Author       : pkq1688
@email        : adolf1321794021@gmail.com
@Date         : 2023-04-20 14:23:34
@LastEditors: pkq1688
@LastEditTime: 2023-08-21 22:04:31
@Description  : 
"""
import random
from pathlib import Path

def handle_base_data(gen_pretrain_data=F):
    file_data_path = Path("/root/autodl-fs/Classical-Modern/双语数据")

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

    if not gen_pretrain_data:
        f_train = open("dataset/classical_cls/cls_data_train.txt","w")
        f_test = open("dataset/classical_cls/cls_data_test.txt","w")
        for gw, bh in zip(gw_list, bh_list):
            if random.random() < 0.99:  
                f_train.write(f"1\t{gw}\n")
                f_train.write(f"0\t{bh}\n")
            else:
                f_test.write(f"1\t{gw}\n")
                f_test.write(f"0\t{bh}\n")
        
        f_train.close()
        f_test.close()
    
    else:
        pass


        
if __name__ == "__main__":
    handle_base_data()