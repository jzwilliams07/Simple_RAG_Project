import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm

class utils:
    """这个类用于存储一系列的RAG工具类
    """    
    def process_file(file_path):
        """_summary_

        Args:
            file_path (str): filepath
        """        
        with open(file_path,encoding="utf-8") as f:
            text = f.read()
            # 将文本都变成一个一个的sentences，这里的实现的sentences是把文本以一行一行的形式分割开，成为sentence
            sentences = text.split('\n')
            return text,sentences
    
    def generate_rag_prompt(data_point):
        """
        Args:
            data_point (dict): 需要填充prompt的各个位置对应的dict
        """        
        return f"""Instruction:
            {data_point["instruction"]}
            ### Input:
            {data_point["input"]}
            ### Response:
        """
    

