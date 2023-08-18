"""
@Author: pkq1688
@Date: 2023-08-18 12:22:41
@LastEditors: pkq1688
@LastEditTime: 2023-08-18 12:23:01
@FilePath: /RewriteLLM/base_model_use.py
@Description: 
"""
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("model/chatglm2-6b-int4", trust_remote_code=True)
model = (
    AutoModel.from_pretrained("model/chatglm2-6b-int4", trust_remote_code=True).half().cuda()
)
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
