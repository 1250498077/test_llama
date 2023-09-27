
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

response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
print(response.text)
