## A pipeline for looking automatically for unreliable off-topic citations in scientific articles

### Introduction
This work aims at assisting and simplifying scientometric scientists' work on finding suspicious citations in scientific articles.

### To do (intergration with codes from Tiziri)
#### Quick mode (only citation context + abstract)
- [ ] Given a list of dois:
1. Automatically get citation context + abstract
2. Get cosine similarity and return the predicted label

(later)
- [ ] Given a folder with pdf files:
1. transfer pdfs to grobid xmls
2. Same as given a list of dois

#### Full text mode (Citation context + whole text of papers)
- [ ] Given a citation context and a folder of cited pdfs:
1. transfer cited pdfs to xmls by grobid
2. compare citation context with the entire text of a paper

### Tutorial
All you need to do is to modify the `eval_config.json` file to provide the evaluation parameters. We have two modes of citation evaluations. 
```json
{
    "context_abstract_evaluation": {
        "models" : "sbert",
        "input_tsv": "dataset.tsv", 
        "output_file": "eval_results.tsv"
    },

    "full_cited_article_evaluation": {
        "models": "sbert",
        "citation_context": "Citation context...",
        "xml_folder": "path to the folder where you store your xml format of cited papers",
        "output_file": "eval_results.tsv"
    }

}
```
The first mode `context_abstract_evaluation` requires a tsv file that is composed of the citation context and the corresponding content (or just abstract) of the cited paper.
The second mode `full_cited_article_evaluation` it's built for evaluating a single citation context by comparing it with its full cited articles. If you only have pdf format of the cited articles, you need to use pdf-to-xml function from *grobid* website to get the xml format (or just use our script if you have grobid installed in your pc), and put these files into a folder.
The third mode is under construction, ideally, given a doi or a pdf file, our script will automatically extract citation contexts and find the matching cited article or content, then, do the similarity evaluation, and returns a file that contains the results.

Use the command:
```bash
python3 main.py --evaluate "eval_config.json"
```
to try our example file
