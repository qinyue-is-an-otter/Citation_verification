# from tqdm import tqdm
import argparse
import json
import Tools.metrics as metrics
import Tools.tools as tools
from os import listdir
import Tools.CC_Abstract as CC_Abstract
import Tools.functions_abstract as functions_abstract

def dois_cited_eval(dois, path):
    # path -> Path to xml folder
    final_df = None
    ids = CC_Abstract.doi_to_pmcid(dois)
    print(ids)
    pmcid = ids[ids.PMCID != 'Not Found']
    p = pmcid.PMCID # Here change
    count = CC_Abstract.fetch_pubmed_articles(p)
    print(f'{count} xml files were downloaded from PMC')
    pmc_citations_df = CC_Abstract.process_files(path)
    print(pmc_citations_df)
    if not pmc_citations_df.empty:
        unique_dois = pmc_citations_df["DOI"].unique()
        final_df = functions_abstract.abstract(unique_dois,pmc_citations_df,ids)
    delete = input('Do you want to delete the xml folder (y/n)? ') # Bug, needs to move it at the end of the process
    if delete == 'y':
        CC_Abstract.shutil.rmtree(path)
    return final_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", help="Enter the JSON filename with evaluation parameters")
    args = parser.parse_args()
    eval_info = args.evaluate

    if (eval_info):
        with open(eval_info, "r", encoding="utf-8") as input_json:
            dict_eval_info = json.load(input_json)
        if dict_eval_info["context_abstract_evaluation"] != {}:
            models, input_tsv, output_file = (dict_eval_info["context_abstract_evaluation"][key] for key in dict_eval_info["context_abstract_evaluation"])
            metrics.evaluate(input_tsv, models, output_file, None) # Can be for model in models
        if dict_eval_info["dois"] != {}:
            models, dois_list_file, output_file = (dict_eval_info["dois"][key] for key in dict_eval_info["dois"])
            if (dois_list_file):
                with open(dois_list_file, "r", encoding="utf-8") as input_doi:
                    doi_list = [doi.strip() for doi in input_doi]
                final_df = dois_cited_eval(doi_list, "try_pipeline")
                metrics.evaluate(None, models, output_file, input_df=final_df)

if __name__ == "__main__":
    main()


