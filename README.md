<img width="1135" alt="image" src="https://github.com/jzwilliams07/Simple_RAG_Project/assets/105477654/a9bb460b-6ab6-4d87-87b3-726f568b6b50"># Simple_RAG_Project
RAG训练营项目，在此仓库更新所有完成的任务
---------
## Task01 simple RAG 实现
使用HuggingFace的transformers库和Meta的FAISS来实现RAG

本地部署模型

### 安装依赖：

新建环境ragPeoject
<img width="673" alt="image" src="https://github.com/jzwilliams07/Simple_RAG_Project/assets/105477654/4ad442c2-9a75-4d32-8073-73ea08b5d335">

在安装中，faiss 1.7.2通过离线安装了faiss-gpu==1.7.2

通过requirements文件安装了其他依赖
<img width="1142" alt="image" src="https://github.com/jzwilliams07/Simple_RAG_Project/assets/105477654/ba1d0957-2f2f-4406-9cf8-fb5a47dead7f">
### Embedding 模型下载， 使用：BAAI/bge-large-zh-v1.5
使用huggingface_cli 通过镜像加速下载bge embedding模型

### 问题：

1. accelerate库安装了后，仍然报错，后发现是因为accelerate的版本太低了，pip install —upgrade后就好了
2. 使用了qwen1.5-7B模型之后，发现生成回答的速度很慢，而且回答的内容是原文档本身的内容，而不是问题的答案，所以重新选用了qwen1.5-4B-chat模型并且apply了chat_template,此时就能得到正确的答案了，现在遗留的问题是为什么使用7B非chat模型回答的内容是文档本身？
<img width="1135" alt="image" src="https://github.com/jzwilliams07/Simple_RAG_Project/assets/105477654/7de40cef-f3ea-4944-b5ae-ad058eef5439">

------------
