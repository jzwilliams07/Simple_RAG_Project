<<<<<<< HEAD
# 国内用户 HuggingFace 高速下载

利用 HuggingFace 官方的下载工具 [huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/download#download-from-the-cli) 和 [hf_transfer](https://github.com/huggingface/hf_transfer) 从 [HuggingFace 镜像站](https://hf-mirror.com/)上对模型和数据集进行高速下载。

## Usage

### 下载模型

从HuggingFace上获取到所需模型名，例如 `lmsys/vicuna-7b-v1.5`：

```bash
python hf_download.py --model lmsys/vicuna-7b-v1.5 --save_dir ./hf_hub
```
**注意事项：**

（1）脚本内置通过 pip 自动安装 huggingface-cli 和 hf_transfer。如果 hf_transfer 版本低于 0.1.4 则不会显示下载进度条，可以手动更新：
```
pip install -U hf-transfer -i https://pypi.org/simple
```

（2）若指定了 `save_dir`，下载过程中会将文件先暂存在 transformers 的默认路径`~/.cache/huggingface/hub`中，下载完成后自动移动到`--save_dir`指定目录下，因此需要在下载前保证默认路径下有足够容量。 

下载完成后使用 transformers 库加载时需要指定保存后的路径，例如：
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="./hf_hub/models--lmsys--vicuna-7b-v1.5")
```
若不指定 `save_dir` 则会下载到默认路径`~/.cache/huggingface/hub`中，这时调用模型可以直接使用模型名称 `lmsys/vicuna-7b-v1.5`。

（3）若不想在调用时使用绝对路径，又不希望将所有模型保存在默认路径下，可以通过**软链接**的方式进行设置，步骤如下：
- 先在任意位置创建目录，作为下载文件的真实存储位置，例如：
    ```bash
    mkdir /data/huggingface_cache
    ```
- 若 transforms 已经在默认位置 `~/.cache/huggingface/hub` 创建了目录，需要先删除：
    ```bash
    rm -r ~/.cache/huggingface
    ```
- 创建软链接指向真实存储目录：
    ```bash
    ln -s /data/huggingface_cache ~/.cache/huggingface
    ``` 
- 之后运行下载脚本时无需指定`save_dir`，会自动下载至第一步创建的目录下：
    ```bash
    python hf_download.py --model lmsys/vicuna-7b-v1.5
    ```
- 通过这种方式，调用模型时可以直接使用模型名称，而不需要使用存储路径：
    ```bash
    from transformers import pipeline
    pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")
    ```

### 下载数据集

和下载模型同理，以 `zh-plus/tiny-imagenet` 为例:
```bash
python hf_download.py --dataset zh-plus/tiny-imagenet --save_dir ./hf_hub
```

### 参数说明
 -  `--model`: huggingface上要下载的模型名称，例如 `--model lmsys/vicuna-7b-v1.5`
 - `--dataset`: huggingface上要下载的数据集名称，例如 `--dataset zh-plus/tiny-imagenet`
 - `--save_dir`: 文件下载后实际的存储路径
 - `--token`: 下载需要登录的模型（Gated Model），例如`meta-llama/Llama-2-7b-hf`时，需要指定hugginface的token，格式为`hf_****`
 - `--use_hf_transfer`: 使用 hf-transfer 进行加速下载，默认开启(True), 若版本低于开启将不显示进度条。
 - `--use_mirror`: 从镜像站 https://hf-mirror.com/ 下载, 默认开启(True), 国内用户建议开启


=======
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
>>>>>>> f0d396c5aac3c46383cb9caef3692f6d11a2a1a3
