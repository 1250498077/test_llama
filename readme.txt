pip install llama-cpp-python

C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64\CL.exe 


模型地址以及下载的地方
Downloading url https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf to path C:\Users\mwh\AppData\Local\llama_index\models\llama-2-13b-chat.Q4_0.gguf


资源
C:\Users\mwh\AppData\Local\llama_index


模型地址
D:\app\llama-2-13b-chat.Q4_0.gguf


https://gpt-index.readthedocs.io/en/stable/examples/llm/llama_2_llama_cpp.html#setup-llm




1.我们能从huggingface获得什么
huggingface的官方网站：http://www.huggingface.co. 在这里主要有以下大家需要的资源。
Datasets：数据集，以及数据集的下载地址
Models：各个预训练模型
course：免费的nlp课程，可惜都是英文的
docs：文档



llama-2-13b-chat.Q4_0.gguf 需要放在c盘



搜索用到的关键字
get_nodes_from_documents
cosine


index.delete_ref_doc("doc_id_0", delete_from_docstore=True)


===================================================
D:\my\神经网络文档\演示程序\week3\llama_index-main\docs\examples\vector_stores\TairIndexDemo.ipynb
### Deleting documents
from llama_index.storage.storage_context import StorageContext

tair_url = (
    "redis://{username}:{password}@r-bp****************.redis.rds.aliyuncs.com:{port}"
)

vector_store = TairVectorStore(
    tair_url=tair_url, index_name="pg_essays", overwrite=True
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)
To delete a document from the index, use `delete` method.
document_id = documents[0].doc_id
document_id
info = vector_store.client.tvs_get_index("pg_essays")
print("Number of documents", int(info["data_count"]))
vector_store.delete(document_id)
info = vector_store.client.tvs_get_index("pg_essays")
print("Number of documents", int(info["data_count"]))


==========================================================================
记录了文档和node节点的管理
D:\my\神经网络文档\演示程序\week3\llama_index-main\docs\core_modules\data_modules\index\document_management.md
