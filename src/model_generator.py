import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm

class GenerativeModel:
    """这是用于生成答案的模型
    """    
    def __init__(self,
                 model_path="Qwen/Qwen1.5-7B",
                 max_input_length = 200,
                 max_generated_length = 200):
        """初始化LLM模型

        Args:
            model_path (str, optional): model_file_path. Defaults to "Qwen/Qwen1.5-7B".
            max_input_length (int, optional): 最长输入token长度. Defaults to 200.
            max_generated_length (int, optional): 最长生成token长度. Defaults to 200.
        """        
        #加载LLM模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map = "auto",
            torch_dtype = torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side = "left",
            add_eos_token = True,
            add_bos_token = True,
            use_fast = False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length = max_input_length
        self.max_generated_length = max_generated_length
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def answer_prompt(self,prompt):
        """生成prompt的答案

        Args:
            prompt (str): 提问LLM的语句

        Returns:
            _type_: _description_
        """        
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )
        #先把prompt进行tokenize
        encoded_input = self.tokenizer([text],
                                       padding=True,
                                       truncation=True,
                                       max_length=self.max_input_length,
                                       return_tensors="pt")
        outputs = self.model.generate(input_ids=encoded_input['input_ids'].to(self.device),
                                      attention_mask = encoded_input['attention_mask'].to(self.device),
                                      max_new_tokens=self.max_generated_length,
                                      do_sample = False)
        decoder_text = self.tokenizer.batch_decode(outputs,
                                                   skip_special_tokens=True)
        return decoder_text
    
