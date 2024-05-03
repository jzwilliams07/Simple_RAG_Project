import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm
class DocumentEmbedder:
    """这个类主要用于对需要知识库的文档进行embedding
    """    
    def __init__(
            self,
            model_name='BAAI/bge-large-zh-v1.5',
            max_length=128,
            max_number_of_sentences=20
    ):
        """初始化embedder

        Args:
            model_name (str, optional):model_path. Defaults to 'BAAI/bge-large-zh-v1.5'.
            max_length (int, optional):maxlen for sentences. Defaults to 128.
            max_number_of_sentences (int, optional): top k. Defaults to 20.
        """        
        #从模型仓库加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        #设置每个sentence的最长token数为多少
        self.max_length = max_length
        #设置需要考虑的最大sentence数量,K
        self.max_number_of_sentences = max_number_of_sentences
    
    def get_document_embeddings(self,sentences):
        """获取文本的embeddings

        Args:
            sentences (str[]):被切分的文本的str数组 

        Returns:
            torch.tensor:返回的是整个文本的平均值
        """        
        #只选取前K个sentences送入GPU
        sentences = sentences[:self.max_number_of_sentences]
        #先进行Tokenize
        encoded_input = self.tokenizer(sentences,
                                       padding=True,
                                       truncation=True,
                                       max_length=128,
                                       return_tensors="pt")
        #计算token的embeddings
        with torch.no_grad():
            model_ouput = self.model(**encoded_input)
        #文本的embedding 应该是里面所有sentences的平均
        #如果文本只有一个embeddding那么这个embedding就是这个文本的embedding
        return torch.mean(model_ouput.pooler_output,dim=0,keepdim=True)