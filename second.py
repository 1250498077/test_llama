
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

# # use Huggingface embeddings
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
# # # create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context, service_context=service_context)


query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
