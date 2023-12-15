
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

from extractionai.config.config import CONFIG_PALM

class Palm_LMM:
    def __init__(self):
        self.config_palm = CONFIG_PALM
        self.init_connexion(self.config_palm["llm_deployment_name"])

    def init_connexion(self,model_name):
        key_path = self.config_palm["path_key"]
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=[self.config_palm["api_auth_scope"]]
        )
        self.authed_session = AuthorizedSession(credentials)
        self.model_name = model_name

    def predict(self, text):
        url = self.config_palm["api_base"]+self.config_palm["project_id"]+self.config_palm["api_localisation"]+self.config_palm["api_localisation"]+self.model_name+":predict"
        data = {
            "instances": [
                {
                    "content": text
                }
            ],
            "parameters": {
                "candidateCount": self.config_palm["candidateCount"],
                "maxOutputTokens": self.config_palm["max_tokens"],
                "temperature": self.config_palm["temperature"],
                "topP": self.config_palm["top_p"],
                "topK": self.config_palm["top_k"]
            }
        }
        response = self.authed_session.request('POST', url,json=data)        
        
        return response.json()
    

class Palm_Embedding(Palm_LMM) :
    def __init__(self):
        self.init_connexion( self.config_palm["embed_deployment_name"])



def test():
    llm = Palm_LMM()
    print("test connexion palm : OK")
    embeddings = Palm_Embedding()
    print("test connexion palm embedding : OK")


if __name__ == "__main__":
    test()