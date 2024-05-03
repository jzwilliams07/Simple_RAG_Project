import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm
from rag_utils import utils
from Custom_Embedder import DocumentEmbedder
from model_generator import GenerativeModel
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_directory",
                        help="The directory that has the documents",
                        default='rag_documents'
                        )
    parser.add_argument("--embedding_model",
                        help="The HuggingFace path to the embedding model to use",
                        default='BAAI/bge-large-zh-v1.5'
                        )
    parser.add_argument("--generative_model",
                        help="The HuggingFace path to the generative model to use",
                        default='Qwen/Qwen1.5-7B'
                        )
    parser.add_argument("--number_of_docs",
                        help="The number of relevant documents to use for context",
                        default=2
                        )
    args = parser.parse_args()
    '''
    把所有目录里的文件都chunk成sentences
    但是注意记录原始的文件地址
    '''
    print('Splitting documents into sentences')
    documents={}
    for idx,file in enumerate(tqdm(os.listdir(args.documents_directory)[:10])):
        current_filepath = os.path.join(args.documents_directory,file)
        #进行chunk
        text,sentences = utils.process_file(current_filepath)
        documents[idx] = {'file_path':file,
                          'sentences':sentences,
                          'document_text':text
                          }
    '''
    现在对所有chunk后的sentences进行embedding
    '''
    print('Getting document embeddings')
    #选择embedder并初始化
    document_embedder = DocumentEmbedder(model_name=args.embedding_model,
                                         max_length=128,
                                         max_number_of_sentences=20
                                         )
    embeddings = []
    #获得embedding
    for idx in tqdm(documents):
        #begin embedidng
        embeddings.append(document_embedder.get_document_embeddings(documents[idx]['sentences']))
    embeddings = torch.concat(embeddings,dim=0).data.cpu().numpy()
    embedding_dimensions = embeddings.shape[1]
    '''
    使用FAISS把所有embeddings建立索引
    '''
    faiss_index = faiss.IndexFlatIP(int(embedding_dimensions))
    faiss_index.add(embeddings)

    #输入你的问题
    question="请你告诉我，香港科技大学（广州） 一期HPC 平台资费问题我应该咨询哪个邮箱？"
    #对问题做embedding
    query_embedding = document_embedder.get_document_embeddings([question])
    distances,indices = faiss_index.search(query_embedding.data.cpu().numpy(),
                                           k=int(args.number_of_docs)
                                           )
    '''
    使用k个最近的文件来为LLM提供上下文来生成answer
    '''
    context=''
    for idx in indices[0]:
        context += documents[idx]['document_text']
    rag_prompt = utils.generate_rag_prompt({'instruction':question,
                                            'input':context})
    '''
    使用LLM生成对应问题的答案，根据提供的context
    '''
    print('Generating answer...')
    generative_model = GenerativeModel(model_path=args.generative_model,
                                       max_input_length=2000,
                                       max_generated_length=2000
                                       )
    answer = generative_model.answer_prompt(rag_prompt)[0].split('### Response:')[1]
    print(answer)
