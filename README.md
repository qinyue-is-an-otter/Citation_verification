## A pipeline to help verify unrelated citations in scientific articles

### Introduction
This work aims at assisting finding suspicious citations in scientific articles.

#### Standard doi mode
- Given a list of dois:
1. Retrieve citation context + abstract from PMC and Crossref
2. Verify citations using indicated methods by the user

#### Citation context + Abstract mode
- Given a tsv file with citation context and cited abstract:
1. Verify citations using indicated methods by the user
2. Calculate the performance of indicated methods with F1 score and accuracy (If the user wishes)

### Tutorial
Make sure you are in Linux system first. After that, you need do to modify the `pipeline_config.json` file to provide the evaluation parameters. Of course, if you wish to use `dois` mode, make sure you have your own key for the PMC database (it's free). We have two modes of citation evaluations. 
```json
{
    "dois": {
        "methods": "sbert,Qwen,Jaccard",
        "doi_list_file": "Input/doi_examples.txt",
        "output_file": "sbert_qwen_jaccard_eval.tsv",
        "key" : "",
        "mail" : ""
    },
    "context_abstract_evaluation": {
        "methods": "sbert,Qwen,Jaccard",
        "doi_list_file": "../Datasets/Input/Annotated_dataset.tsv",
        "output_file": "../Datasets/Annotated_dataset_eval.tsv",
        "method_accuracy_check": "Yes"
    }
}
```
The first mode `dois` it's built for scanning through a list of given DOIs to detect possible unrelated citations. Given a doi or a pdf file, our script will automatically extract citation contexts and find the matching cited article or content, then, does the prediction with indicated methods, and returns a file that contains the results.

The second mode `context_abstract_evaluation` requires a tsv file that is composed of the citation context and the corresponding content (or just abstract) of the cited paper. If the `method_accuracy_check` is *Yes*, then, there must be a column called *Label* in the given file.

Users can modify different citation assessment methods in the `methods`. The available values are: *Jaccard,Rouge,bert,distilbert,sbert,T5,Qwen,Mistral*. If you want to use Mistral, you are obligated to log into the huggingface-hub.

Use command to build the virtual environment under Linux
```bash
conda env create -f environment.yml # Create the environment
conda activate citation_verif # activate the environment
```
If you don't want to run the program on GPU, you can simply use pip to install the packages from the `requirements.txt` file. (Running Qwen and Mistral is not recommended for this environment)

Move the terminal to the Pipeline folder, and use the command:
```bash
python3 main.py --evaluate "pipeline_config.json"
```
to try our example files with both modes.


