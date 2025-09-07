## A pipeline to help verify unrelated citations in scientific articles

### Introduction
This work aims at assisting finding suspicious citations in scientific articles.

#### Citation context + Abstract mode
- Given a tsv file with citation context and cited abstract:
1. Verify citations using indicated methods by the user
2. Calculate the performance of indicated methods with F1 score and accuracy (If the user wishes)

#### Standard doi mode
- Given a list of dois:
1. Retrieve citation context + abstract from PMC and Crossref
2. Verify citations using indicated methods by the user

### Tutorial
All you need to do is to modify the `eval_config.json` file to provide the evaluation parameters. We have two modes of citation evaluations. 
```json
{
    "context_abstract_evaluation": {
        "models": "sbert,Qwen,Jaccard",
        "doi_list_file": "../Datasets/Input/Annotated_dataset.tsv",
        "output_file": "../Datasets/Annotated_dataset_eval.tsv",
        "method_accuracy_check": "Yes"
    },
    "dois": {
        "models": "sbert,Qwen,Jaccard",
        "doi_list_file": "Input/doi_examples.txt",
        "output_file": "sbert_qwen_jaccard_eval.tsv"
    }
}
```
The first mode `context_abstract_evaluation` requires a tsv file that is composed of the citation context and the corresponding content (or just abstract) of the cited paper. If the `method_accuracy_check` is *Yes*, then, there must be a column called *Label* in the given file.

The second mode `dois` it's built for scanning through a list of given DOIs to detect possible unrelated citations. Given a doi or a pdf file, our script will automatically extract citation contexts and find the matching cited article or content, then, does the prediction with indicated methods, and returns a file that contains the results.

Use command
```bash
conda env create -f environment.yml
```
to create the environment

Move the terminal to the Pipeline folder, and use the command:
```bash
python3 main.py --evaluate "pipeline_config.json"
```
to try our example files with both modes.


