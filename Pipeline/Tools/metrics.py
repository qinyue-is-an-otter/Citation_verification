
import Tools.tools as tools
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel,DistilBertModel,T5Model,AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

def get_rouge_score(citation_context, abstract):
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True) # rouge1 means n-gram = 1, rouge2 exists as well
    score = scorer.score(citation_context, abstract)
    return score['rouge2'][0]

def jaccard_score(phrase1, phrase2, stop_words):
    phrase1_set = tools.clean_stopwords_set(phrase1, stop_words)
    phrase2_set = tools.clean_stopwords_set(phrase2, stop_words)
    intersection_set = phrase1_set & phrase2_set
    union = phrase1_set | phrase2_set
    return len(intersection_set) / len(union)

def load_Bert_family(model_name):
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True )
    elif model_name == "distilbert":
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    elif model_name == "sbert":
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    # Needs pad token
    elif model_name == "T5":
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        model = T5Model.from_pretrained("google-t5/t5-small").encoder
    else:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    return model, tokenizer

def tokenize(tokenizer, sentence):
    return tokenizer(sentence, padding='max_length', max_length=512, return_tensors='pt', truncation=True)

def refine_for_sentence_embedding(sentence, model, tokenizer, device):
    tokens = tokenize(tokenizer, sentence)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    outputs = model(input_ids = input_ids, attention_mask = attention_mask)
    last_hidden_states = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    masked_embedding = mask * last_hidden_states

    # Get an average of the whole sentence embedding
    sum = torch.sum(masked_embedding,1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9) # avoid 0
    sentence_embedding = sum/sum_mask

    return sentence_embedding

def results_bert_family(model, tokenizer, model_name, df_test, col_name, context):
    """
    model: The language model (object)
    df_test: Dataframe of the input file (dataframe)
    col_name: The column name, title in cited paper, abstract etc (string)
    file_out: Output filename (string)
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    model.eval()
    with torch.no_grad():
        new_col = "Cosine_similarity_" + model_name
        df_test[new_col] = ""
        for id in tqdm(df_test.index):
            df = df_test.loc[[id]]
            cited_abstract = df[col_name].values[0]
            cited_abstract = tools.clean_abstract(cited_abstract)
            if (cited_abstract != "nan"):
                cited_abstract = " ".join(cited_abstract.split()) # Reduce the spaces
                citation_context = df[context].values[0]
                emb1 = refine_for_sentence_embedding(citation_context, model, tokenizer, device).cpu().detach().numpy()
                emb2 = refine_for_sentence_embedding(cited_abstract, model, tokenizer, device).cpu().detach().numpy()
                similarity = cosine_similarity(emb1, emb2)[0][0]
                df_test.at[id, new_col] = str(similarity)
                df_test.at[id, col_name] = cited_abstract
    return df_test


def make_file_with_scores(file_input, methods, input_csv):
    
    if (file_input):
        df_new = tools.get_cleaned_df(file_input)# .sample(10000)
    else:
        df_new = input_csv
    for method in methods:
        if (method == "Qwen" or method == "Mistral"):
            df_new = qwen(method, df_new)
        elif (method == "Rouge" or method == "Jaccard"):
            for i in tqdm(df_new.index):
                citation_context = df_new['Citation_context'][i]
                abstract = df_new['Cited_content'][i]
                if method == "Rouge":
                    if "Rouge_score" not in df_new.columns:
                        rouge = get_rouge_score(citation_context, abstract)
                        df_new["Rouge_score"] = rouge
                    else:
                        rouge = get_rouge_score(citation_context, abstract)
                        df_new.at[i, "Rouge_score"] = rouge
                if method == "Jaccard":
                    if "Jaccard_score" not in df_new.columns:
                        jaccard = jaccard_score(citation_context, abstract, tools.stopwords.words('english'))
                        df_new["Jaccard_score"] = jaccard
                    else:
                        jaccard = jaccard_score(citation_context, abstract, tools.stopwords.words('english'))
                        df_new.at[i, "Jaccard_score"] = jaccard
        else:
            model, tokenizer = load_Bert_family(method)
            df_new = results_bert_family(model, tokenizer, method, df_new, "Cited_content", "Citation_context")
    return df_new

# For prompting approach
def user_prompt_evaluation(abstract, context):
    return f'''
    The abstract is: {abstract},
    The citation context is: {context}
    Now please evaluate whether citation context and the cited abstract is related or not.
    '''
    
def qwen(method,df):
    if (method == "Qwen"):
        model_name = "Qwen/Qwen3-8B" # Here you can use other size
    else:
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    for id in tqdm(df.index):
        citation_context = df.loc[id, "Citation_context"]
        cited_content = df.loc[id, "Cited_content"]
        messages = [
            {"role": "system", "content": "You only answer 'Related' or 'Unrelated'"},
            {"role": "user", "content": user_prompt_evaluation(cited_content, citation_context)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        use_cuda = torch.cuda.is_available()
        device = ("cuda" if use_cuda else "cpu")
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        df.loc[id, f"{method}_prediction"] = content
    return df
    


dict_threshold = {
    "sbert" : 0.18, # Recommended threshold range for sbert: 0.16 - 0.19
    "bert" : 0.754,
    "T5" : 0.741,
    "distilbert" : 0.804,
    "roberta" : 0.973,
    "gpt2" : 0.998,
    "Bleu" : 7.73e-155,
    "Rouge" : 0.001,
    "Jaccard" : 0.0275
}

def score_prediction(methods, df, dict_threshold):
    for method in methods:
        if (method == "Qwen" or method == "Mistral"):
            continue
        else: 
            threshold = dict_threshold[method] # column name of the score
            prediction = f"{method}_prediction"
            if (method == "Rouge" or method == "Jaccard"):
                method_column = method + "_score"
            else:
                method_column = "Cosine_similarity_" + method
            df[prediction] = ""
            for i in tqdm(df.index):

                if float(df[method_column][i]) >= threshold:
                    df.at[i, prediction] = "Related"
                else:
                    df.at[i, prediction] = "Unrelated"
    return df

def method_evaluation(df, method):
    prediction = f"{method}_prediction"
    tp = df[(df["Label"] == "Related") & (df[prediction] == "Related")].shape[0]
    tn = df[(df["Label"] == "Unrelated") & (df[prediction] == "Unrelated")].shape[0]
    fp = df[(df["Label"] == "Unrelated") & (df[prediction] == "Related")].shape[0]
    fn = df[(df["Label"] == "Related") & (df[prediction] == "Unrelated")].shape[0]
    # print(tp,tn,fp,fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn != 0 else 0
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0 # tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return f"For {method}, the f1 score is: {f1}, the precision is: {precision}, the recall is: {recall}, the accuracy is {accuracy}\n"


def evaluate(input_file, methods, output_file, input_df=None):
    '''
    Input_file: a .tsv file
    input_df: a dataframe
    '''
    methods = methods.split(',')
    df_prediction = make_file_with_scores(input_file, methods, input_df) # with only prediction scores
    df_new = score_prediction(methods, df_prediction, dict_threshold) # verify if the prediction is good or not
    df_new.to_csv(output_file, sep='\t', encoding="utf-8", index = False)
    return df_new