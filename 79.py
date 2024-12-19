from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter,TokenTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os
import gc
import re


DOCS_DIR = 'data/A_document'
EMB_MODEL = 'BAAI/bge-large-zh-v1.5'
RERANK_MODEL = "BAAI/bge-reranker-large"
PERSIST_DIR = './vectordb' 
QUERY_DIR = 'data/A_question.csv'
SUB_DIR = 'data/submit_example.csv'

query = pd.read_csv(QUERY_DIR)
sub = pd.read_csv("data/submit_example.csv")

loader = PyPDFDirectoryLoader(DOCS_DIR)
pages = loader.load_and_split()
pdf_list = os.listdir(DOCS_DIR)

pdf_text = { pdf_page.metadata['source'][-8:]:'' for pdf_page  in pages }
for pdf in tqdm(pdf_list):
    for pdf_page in pages:
        if pdf in pdf_page.metadata['source']:
            pdf_text[pdf] += pdf_page.page_content
        else:
            continue


def filter_text(text):
    # 页码清除 效果不好
#     page_id_pattern1 = r'\n\d+\s*/\s*\d+\s*\n'
#     page_id_pattern2 = r'\n\d+\n'
#     page_id_pattern3 = r'\n\d+\s*?'

#     page_id_pattern = page_id_pattern1+'|'+page_id_pattern2+'|'+page_id_pattern3
#     text = re.sub(page_id_pattern,'',text)
    
    # '\n', ' ' 删除
    text = text.replace('\n','').replace(' ','')
    
    # 删除页码
    
    # 删除本文档为2024CCFBDC***
    head_pattern = '本文档为2024CCFBDCI比赛用语料的一部分。[^\s]+仅允许在本次比赛中使用。'
    # news_pattern
    pattern1 = r"发布时间：[^\s]+发布人：新闻宣传中心"
    pattern2 = r"发布时间：[^\s]+发布人：新闻发布人"
    pattern3 =  r'发布时间：\d{4}年\d{1,2}月\d{1,2}日'
    news_pattern = head_pattern+'|'+pattern1+'|'+pattern2+'|'+pattern3
    text = re.sub(news_pattern,'',text)
    
    
    # report_pattern
    report_pattern1 = '第一节重要提示[^\s]+本次利润分配方案尚需提交本公司股东大会审议。'
    report_pattern12 = '一重要提示[^\s]+股东大会审议。'
    report_pattern13 = '一、重要提示[^\s]+季度报告未经审计。'
    report_pattern2 = '本公司董事会及全体董事保证本公告内容不存在任何虚假记载、[^\s]+季度财务报表是否经审计□是√否'
    report_pattern3 = '中国联合网络通信股份有限公司（简称“公司”）董事会审计委员会根据相关法律法规、[^\s]+汇报如下：'
    report_pattern = report_pattern1+'|'+report_pattern12+'|'+report_pattern13+'|'+report_pattern2+'|'+report_pattern3
    text = re.sub( report_pattern,'',text)
#     white paper 版本一 效果不好
    # 优先级别 bp1 bp2 bp3
#     bp_pattern_law = '版权声明[^\s]+追究其相关法律责任。'
#     bp_pattern1 = r'目录.*?披露发展报告（\d{4}年）' # 只针对AZ08.pdf
#     bp_pattern2 = r'目录.*?白皮书.*?（\d{4}年）'
#     bp_pattern3 = r'目录.*?白皮书'
#     bp_pattern = bp_pattern_law  +'|'+bp_pattern1+'|'+bp_pattern2+'|'+bp_pattern3
#     text = re.sub(bp_pattern,'',text)
    
#     print(text)
    
    return text
    



for pdf_id in pdf_text.keys():
    pdf_text[pdf_id] = filter_text(pdf_text[pdf_id])
with open('AZ.txt','w',encoding = 'utf-8') as file:
    pdf_all = ''.join(list(pdf_text.values())).encode('utf-8', 'replace').decode('utf-8')
    file.write( pdf_all)    



from langchain_community.document_loaders import TextLoader
loader = TextLoader("AZ.txt",encoding="utf-8")
documents = loader.load()
#分割文本
text_splitter = RecursiveCharacterTextSplitter(        
        chunk_size=245,             
        chunk_overlap=128,
        separators = ["。", "！", "？"],
        keep_separator='end',
    )
docs = text_splitter.split_documents(documents)



embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL, show_progress=True)
vectordb = FAISS.from_documents(   
    documents=docs,
    embedding=embeddings,
)

vectordb.save_local(PERSIST_DIR)



import jieba
dense_retriever = vectordb.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(
    docs, 
    k=5, 
    bm25_params={"k1": 1.5, "b": 0.75}, 
    preprocess_func=jieba.lcut
)
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6])



from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def rerank(questions, retriever, top_n=1, cut_len=384):
    rerank_model = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    compressor = CrossEncoderReranker(model=rerank_model, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    rerank_answers = []
    for question in tqdm(questions):
        relevant_docs = compression_retriever.invoke(question)
        answer=''
        for rd in relevant_docs:
            answer += rd.page_content
        rerank_answers.append(answer[:245])
    return rerank_answers

questions = list(query['question'].values)
rerank_answers = rerank(questions, ensemble_retriever)
print(rerank_answers[0])


def emb(answers, emb_batch_size = 4):
    model = SentenceTransformer(EMB_MODEL, trust_remote_code=True)
    all_sentence_embeddings = []
    for i in tqdm(range(0, len(answers), emb_batch_size), desc="embedding sentences"):
        batch_sentences = answers[i:i+emb_batch_size]
        sentence_embeddings = model.encode(batch_sentences, normalize_embeddings=True)
        all_sentence_embeddings.append(sentence_embeddings)
    all_sentence_embeddings = np.concatenate(all_sentence_embeddings, axis=0)
    print('emb_model max_seq_length: ', model.max_seq_length)
    print('emb_model embeddings_shape: ', all_sentence_embeddings.shape[-1])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return all_sentence_embeddings

all_sentence_embeddings = emb(rerank_answers)


sub['answer'] = rerank_answers
sub['embedding']= [','.join([str(a) for a in all_sentence_embeddings[i]]) for i in range(len(all_sentence_embeddings))]
sub.to_csv('submit.csv', index=None)
sub.head()
