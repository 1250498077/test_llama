
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
import ast
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import StorageContext, load_index_from_storage


from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser

import json

text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
llm = LlamaCPP(
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="C:/Users/Public/llama-2-13b-chat.Q4_0.gguf"
)



documents = SimpleDirectoryReader('D:/my/神经网络文档/演示程序/test/data').load_data()
# index = VectorStoreIndex.from_documents(documents)
node_parser = SimpleNodeParser.from_defaults(
    text_splitter=text_splitter,
)
nodes = node_parser.get_nodes_from_documents(documents)
# print("nodes", nodes)
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
# # # create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# index = VectorStoreIndex(nodes, service_context=service_context)
# index.storage_context.persist()



storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, service_context=service_context)
retriever = index.as_retriever(similarity_top_k=1)
nodes = retriever.retrieve("AI was in the air in the mid 1980s, but there were two things especially that made ")
# parsed_data = ast.literal_eval(nodes)
# 此时parsed_data包含了Python对象的结构

for i, node in enumerate(nodes):
    # print(node)
    print(node.score)
    print(node.metadata)
    print(node.text)
    print(node.node_id)
# print("nodes========================================", nodes.json())

# 使用json.dumps()方法将字典转换为JSON字符串
# json_string = json.dumps(nodes)


