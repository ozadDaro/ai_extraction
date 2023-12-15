import sys
sys.path.append("..")
from searchengine.model.llm import LLM
from searchengine.database.db import DB 
from searchengine.seo.se import SearchEngineLCL
from searchengine.tools.utils import process_output
from searchengine.config.config import CONFIG_OPENAI, CONFIG_PIPELINE

def main():
    model = LLM("openai",CONFIG_PIPELINE )
    db = DB(model, CONFIG_PIPELINE, db_name="chromadb_hybride")
    se = SearchEngineLCL(db, model)
    print("Querying")
    if  CONFIG_PIPELINE["process_mode"]=="display":
        query = 'start'
        while query != 'Q':
            query = input("Posez votre question, sinon tappez 'Q' pour quiter : \n")
            if query != 'Q':
                query += " (réponse en français)"
                result, sources, score_sources = se.local_query_with_sources(query,
                                                     top_k=CONFIG_PIPELINE["chunk_topK"], 
                                                     reordering_loss_in_midel= CONFIG_PIPELINE["reordering_loss_in_midel"])
                process_output(result, sources, score_sources, verbose=False)
            else:
                print("\n Au revoir \n")


if __name__ == "__main__":
    main()