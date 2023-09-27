
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import StorageContext, load_index_from_storage
llm = LlamaCPP(
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="C:/Users/Public/llama-2-13b-chat.Q4_0.gguf"
)


# use Huggingface embeddings
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
# # create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)
filename_fn = lambda filename: {'file_name': filename}
# load documents
documents = SimpleDirectoryReader(
    # "D:/my/神经网络文档/演示程序/test/data",
    "./data",
    # 给每一个节点都设置文件名
    file_metadata=filename_fn
).load_data()
# documents = SimpleDirectoryReader('./data').load_data()
# # create vector store index
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist()



# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)



# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# embeddings = model.encode(sentences)
# print(embeddings)