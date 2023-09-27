
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import StorageContext, load_index_from_storage


from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser

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
# 加载缓存里面的index 
index = load_index_from_storage(storage_context, service_context=service_context)


index.delete_ref_doc("799cb399-43f4-436c-9a39-16d025e502fb", delete_from_docstore=True)
index.storage_context.persist()
# for doc in documents:
#     index.insert(document=doc)


# index.storage_context.persist()
# query_engine = index.as_query_engine()



# retriever = index.as_retriever(similarity_top_k=1)
# nodes = retriever.retrieve("What did the author do growing up?")
# print("nodes========================================", nodes)
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)

# response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
# print(response.text)


# # # use Huggingface embeddings
# embed_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )
# # # # create a service context
# service_context = ServiceContext.from_defaults(
#     llm=llm,
#     embed_model=embed_model,
# )
# # load documents
# documents = SimpleDirectoryReader(
#     "D:/my/神经网络文档/演示程序/test"
# ).load_data()
# # create vector store index
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)
# set up query engine 这行代码默认会在当前目录的 ./storage 文件夹下
# 


# rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
# index = load_index_from_storage(storage_context, service_context=service_context)


# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)



# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# embeddings = model.encode(sentences)
# print(embeddings)s