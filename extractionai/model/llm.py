import openai
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings

from extractionai.config.config import OPENAI_CONFIG
from extractionai.utils.utils import init_openai
from extractionai.model.palm import Palm_LMM, Palm_Embedding


class LLM:
    def __init__(self, model_name, config_llm):
        if "palm" in model_name:
            self.llm = Palm_LMM()
            self.embeddings = Palm_Embedding()


        elif "openai" in model_name :

            
            init_openai(CONFIG_OPENAI)
            self.config = CONFIG_OPENAI
            print("..........>",self.config)
            self.llm = AzureOpenAI(deployment_name=self.config["llm_deployment_name"], 
                                    openai_api_key=openai.api_key, 
                                    openai_api_version=openai.api_version,
                                    temperature=0.0)
            self.embeddings = OpenAIEmbeddings(
                deployment=self.config["embed_deployment_name"],
                openai_api_base=openai.api_base,
                openai_api_type=openai.api_type,
                openai_api_version=openai.api_version,
                openai_api_key=openai.api_key,
                chunk_size=self.config["embed_chunk_size"]                
            )
            
            
        else:
            pass