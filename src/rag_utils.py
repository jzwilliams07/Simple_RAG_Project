import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pdfplumber
from openpyxl import Workbook
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
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
class PDFProcessor:
    def __init__(self, input_folder, output_folder, max_workers=1):#切勿使用多线程，使用1个worker的速度反而是最快的
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.max_workers = max_workers
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def process_pdf(self, filepath):
        """处理单个PDF文件，提取文本和表格。

        Args:
            filepath (str): 完整的PDF文件路径。
        """
        text_output_path = os.path.join(self.output_folder, os.path.basename(filepath).replace('.pdf', '_text.txt'))
        table_output_path = os.path.join(self.output_folder, os.path.basename(filepath).replace('.pdf', '_tables.txt'))
        
        # 打开PDF文件一次
        with pdfplumber.open(filepath) as pdf:
            text_content, table_content = self.extract_all_content(pdf)
            self.write_text(text_output_path, text_content)
            self.write_tables(table_output_path, table_content)
            
    def extract_all_content(self, pdf):
        """从PDF中提取所有文本和表格内容，存储在内存中，等待一次性写入."""
        text_content = []
        table_content = []
        for page in pdf.pages:
            # 提取文本
            textdata = page.extract_text()
            if textdata:
                text_content.append(textdata)
            # 提取表格
            for table in page.extract_tables():
                formatted_table = ["\t".join([str(cell) if cell else "" for cell in row]) for row in table]
                table_content.append(formatted_table)
        return text_content, table_content

    def write_text(self, outputfile, text_content):
        """将所有提取的文本内容写入文件."""
        with open(outputfile, 'w', encoding='utf-8') as out_file:
            for text in text_content:
                out_file.write(text + "\n")

    def write_tables(self, outputfile, table_content):
        """将所有提取的表格内容写入文件."""
        with open(outputfile, 'w', encoding='utf-8') as out_file:
            for table in table_content:
                out_file.write("\n".join(table) + "\n\n")

    def create_excel(self, table_content, outputfile):
        """将提取的表格内容保存到Excel文件."""
        workbook = Workbook()
        sheet = workbook.active
        for table in table_content:
            for row in table:
                sheet.append(row.split('\t'))
        workbook.save(filename=outputfile)

    def process_all_pdfs(self):
        """使用多线程处理文件夹中的所有PDF文件，并显示进度条。"""
        pdf_files = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith('.pdf')]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.process_pdf, pdf_files), total=len(pdf_files), desc='Processing PDFs'))
