import pandas as pd
#import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

regex = r"&[^\&\;]*;|<[^\<\>]*>" # Get the content like </jats:sec>, &lt; and delete it

# Methods for removing stopwords, returns a set object
def clean_stopwords_set(phrase, stop_words):
    phrase = re.sub(r"[^a-zA-Z \d]", " ", phrase) # Replace all the ponctuations by " "
    set_phrase = set(phrase.split(" ")).difference(stop_words)
    return set_phrase
# Returns a string
def clean_stopwords_list(phrase, stop_words):
    phrase = re.sub(r"[^a-zA-Z \d]", " ", phrase) # Replace all the ponctuations by " "
    list_words = phrase.split(" ")
    cleaned_phrase = []
    for word in list_words:
        if word not in stop_words:
            cleaned_phrase.append(word)
    return " ".join(cleaned_phrase)

def jaccard_score(phrase1, phrase2, stop_words = stopwords):
    phrase1_set = clean_stopwords_set(phrase1, stop_words)
    phrase2_set = clean_stopwords_set(phrase2, stop_words)

    intersection_set = phrase1_set & phrase2_set
    union = phrase1_set | phrase2_set
    return len(intersection_set) / len(union)


# ------------------------------------------ Filters for the validated dataset ---------------------------

# Read dataframe from a file
def get_cleaned_df(file):
    df = pd.read_csv(file, sep='\t', encoding="utf-8", on_bad_lines='skip').dropna()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

# Exclude empty abstract
def abstract_available(df):
    df_new = df.dropna()
    print(f'We extracted {len(df)} citations by default\nNow {len(df_new)} citations are useful as we can extract their cited abstract')
    return df_new

# Get cleaned abstract text
def clean_abstract(abstract):
    abstract = abstract.replace("\n"," ")
    abstract = re.sub(regex, "", abstract)
    return abstract

# Exclude useless citation context (Less than 3 charactors)
def citation_context_available(df):
    df_new = df
    df_new = df_new[df_new["Citation_context"].map(str).map(len) > 2]
    print(f'We extracted {len(df)} citations by default\nNow {len(df_new)} citations are useful, as the citation context is longer than 2 characters')
    return df_new

# Exclude useless citation context and empty abstract
def abstract_citation_context_available(df):
    df_new = df.dropna()
    df_new = df_new[df_new["Citation_context"].map(str).map(len) > 2]
    print(f'We extracted {len(df)} citations by default\nNow {len(df_new)} citations are useful')
    return df_new

# Exclude citation context and abstract that has less than 'word_count' words
def assessment_available(df, word_count):
    map_split = lambda phrase: phrase.split(' ')
    extra_space_exclude = lambda phrase: " ".join(phrase.split()) # exclude extra spaces within a sentence
    df_new = df.dropna()
    df_new = df_new[df_new["Citation_context"].map(str).map(extra_space_exclude).map(map_split).map(len) > word_count]
    df_new = df_new[df_new["Cited_content"].map(str).map(extra_space_exclude).map(map_split).map(len) > word_count]
    # print(f'We extracted {len(df)} citations by default\nNow {len(df_new)} citations are useful')
    return df_new

def dict_to_df(input_dict):
    df = pd.DataFrame.from_dict(input_dict)
    return df

# Test
#df = pd.read_csv("Results/The_lancet_2022.csv", sep='\t', encoding='utf-8')
#df_new = assessment_available(df, 2) 
#print(df_new.head(30)["Citation_context"])



# ----------------------------------------- Similar abstract (citation_context + abstract) ---------------

# Check if the first phrase has more than 'diff_elem_num' of words that are different from phrase2
def similar(phrase1, phrase2, diff_elem_num):
    '''
    If the different set has more than 'diff_elem_num' elements, then, return False
    '''
    phrase1 = re.sub(r"[^a-zA-Z \d]", " ", phrase1) # Replace all the ponctuations by " "
    phrase2 = re.sub(r"[^a-zA-Z \d]", " ", phrase2)
    set1 = set(phrase1.split(" "))
    set2 = set(phrase2.split(" "))
    diff_set = set1.difference(set2) # The words in set1 that didn't appear in set2
    if len(diff_set) < diff_elem_num:
        return True
    return False

# This function compares a single phrase with all the other phrases in a dataframe
def verify_duplicated_abstract(df, info_phrase1, phrase2_index, df_rows, dict_duplicated):
    # Verify if the same Abstract has different dois within the file
    if (phrase2_index >= df_rows):
        return dict_duplicated
    else:
        phrase2 = df['Abstract'][phrase2_index]
        doi_phrase2 = df['Doi'][phrase2_index]

        phrase1 = info_phrase1['Abstract'].values[0]
        title_phrase1 = info_phrase1['Title'].values[0]
        doi_phrase1 = info_phrase1['Doi'].values[0]
        if similar(phrase1, phrase2, 5): # If ducplicated phrase exist
            if title_phrase1 in dict_duplicated:
                dict_duplicated[title_phrase1].append(doi_phrase2)
            else:
                dict_duplicated[title_phrase1] = [doi_phrase1, doi_phrase2]
            #print(phrase2_index)
        return verify_duplicated_abstract(df, info_phrase1, phrase2_index + 1, df_rows, dict_duplicated)

# Function to get possible suspicious abstract from the dictionary of the duplicated abstract.
# dict_duplicated: {title:[doi1, doi2, ... doin]}
def get_suspicious_abstract(dict_duplicated):
    suspicious_df = {}
    for key in dict_duplicated:
        doi_list = dict_duplicated[key]
        if len(doi_list) < 2:
            print("Doi not duplicated")
        else:
            first_doi = set(doi_list[0].replace('.','/').split('/'))
            for doi in doi_list[1:]:
                other_doi = set(doi.replace('.','/').split('/'))
                diff_set = first_doi.symmetric_difference(other_doi)
                # print(diff_set)
                for elem in diff_set:
                    if re.match(r'v\d', elem) == None: # filter the preprints to do:
                        suspicious_df[key] = doi_list
    return suspicious_df

# Function to list all duplicated (will mark the suspicious ones) abstracts.
# Takes in an input file, gives an output
def Get_all_duplicated_abstract(file, output):
    df = get_cleaned_df(file)
    #print(df)
    #return 0
    df_rows = len(df.index)
    dict_duplicated = {}

    # Search duplicated abstract for each phrase
    for i in df.index:
        info_phrase1 = df.loc[[i]]
        phrase2_index = i + 1
        dict_duplicated = verify_duplicated_abstract(df, info_phrase1, phrase2_index, df_rows, dict_duplicated)

    dict_suspicious = get_suspicious_abstract(dict_duplicated)
    with open(output, "w", encoding='utf-8') as file_out:
        file_out.write(f'Title\tDois\tSuspicious\n')

        for key in dict_duplicated:
            # Key is the title
            suspicious = "No"
            if key in dict_suspicious:
                suspicious = "Yes"
            dois = ", ".join(dict_duplicated[key])
            file_out.write(f'{key}\t{dois}\t{suspicious}\n')

    return dict_duplicated, dict_suspicious

#Get_all_duplicated_abstract("../Paper_mill_simulation/abs_list_dup.tsv", "try.csv")


# ----------------------------------------- Merge all data ---------------------------------------

# The files in the directory "Results" are the most important ones
import glob
from json import loads, dumps, load

def merge_csv(folder_path, output_file, format):
    df = pd.DataFrame()
    file_list = glob.glob(f'{folder_path}/*.{format}')
    for file in file_list:
        if file.endswith(format):
            df_file = assessment_available(get_cleaned_df(file), 5)
            df = pd.concat([df, df_file], ignore_index=True)
    df.to_csv(output_file, sep="\t", encoding="utf-8", index=False)
    return df

def df_to_json(df, output_file):
    df.index = [f'Citation {str(i)}' for i in range(len(df.index))]
    json_result = df.to_json(orient="index")
    parsed = loads(json_result)
    with open(output_file, "w", encoding="utf-8") as output_json:
        output_json.write(dumps(parsed, indent=4))
    # dumps(parsed, output_file, indent=4)

def read_json(input_file):
    with open(input_file, "r", encoding='utf-8') as try_read:
        data = load(try_read)
    return data

# combined_df = merge_csv("Results", "Assement_available_citations.tsv","csv")
# print(combined_df)
# df_to_json(combined_df, "Assement_available_citations.json")
#print(dumps(read_json("try.json"), indent = 4))

def add_click_link(inputfile):
    df = get_cleaned_df(inputfile)
    df["Cited_paper_title"] = df["Cited_paper_title"].apply(lambda title:'=HYPERLINK("https://www.google.com/search?q='+title+'")')
    df.to_csv("Verify_cited_abstract.tsv", sep='\t', encoding='utf-8', index=False)

#add_click_link("Assement_available_citations.tsv")

def annotate_labels(citation_type, df):
    if (citation_type == "Reliable"):
        df["Label"] = ["Reliable" for _ in df.index]
    elif(citation_type == "Unreliable"):
        df["Label"] = ["Unreliable" for _ in df.index]
    return df

def fusion_dataset(df_good, df_generated, new_csv, new_json):
    result = pd.concat([df_good, df_generated], join="inner")
    if (new_csv):
        result.to_csv(new_csv, sep="\t", encoding="utf-8", index=False)
    if (new_json):
        df_to_json(result, new_json)
    return result

def try_fusion(csv_good, csv_generated):
    df_good = assessment_available(get_cleaned_df(csv_good), 5)
    df_good = annotate_labels("Reliable", df_good)
    df_generated = assessment_available(get_cleaned_df(csv_generated), 5)
    df_generated = annotate_labels("Unreliable", df_generated)
    new_df = fusion_dataset(df_good, df_generated, new_csv="Olessya_filter.tsv", new_json=None)
    print(new_df.shape[0])

# combined_df = merge_csv("Results", "raw_dataset.tsv", "csv")
#df_to_json(combined_df, "Main_dataset.json")
# try_fusion("Main_dataset.tsv", "../Paper_mill_simulation/Generated_citations.tsv")

# ----------------------------------- Get citation context to generate bad citations -----------------
def get_citation_contexts(input, output):
    df_clean = get_cleaned_df(input)
    df_clean = assessment_available(df_clean, 5)["Citation_context"]
    df_drop_dup = df_clean.drop_duplicates()
    df_drop_dup.to_csv(output, sep="\t", encoding='utf-8', index=False)

# get_citation_contexts("Results/Cell_2018.csv", "good_contexts.tsv")