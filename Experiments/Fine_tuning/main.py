#from nltk.corpus import stopwords
#import sys
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import *
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

# Local classes
from Dataset import Dataset
from My_Classifier import My_Classifier

import logging
logging.disable(logging.WARNING)
#logging.set_verbosity_error() # Eliminate useless logs from using models
# import wandb

# wandb.init(
#     project="bert_cos_simi",
#     config={
#         "model": "bert",
#     }
# )


def load_models(model_name):
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True )
    else:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    return model, tokenizer


def load_dataset(input_file, samples=1000):
    df = pd.read_csv(input_file, sep = "\t", encoding = "utf-8")
    df_unre = df[df["Label"] == "Unrelated"].sample(samples)
    df_rela = df[df["Label"] == "Related"].sample(samples)
    #print(df_rela.head())
    df = pd.concat([df_rela, df_unre])
    np.random.seed(123)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state = 666), [int(0.8 * len(df)), int(0.9 * len(df))]) # frac = 0.5 means 50% of the data
    # print(len(df_train[df_train["Label"]=="Reliable"]), len(df_val), len(df_test))
    return df_train, df_val, df_test


def train(tokenizer, model, train_data, val_data, learning_rate, epochs, batch_size):
  # Get different sets with the help of Dataset class we defined
    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer)
    # DataLoader gets data according to batch_size
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
  # Do we use GPU?
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Define loss function (CrossEntropy is ideal for classification tasks) and optimization stragegy
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # For freezing parameters -----------------------------------------
    for name, param in model.model.named_parameters():
        param.requires_grad = False

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # Start training~~
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            step = 0
            # tqdm is only here for showing progress
            for train_input, train_label, train_sample in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device) # for bert
                #input_id = train_input['input_ids'].to(device)

                output = model(input_id, mask) # Here the output is all the layers we used in class "BertClassifier"
                # Calculate the loss
                train_loss = criterion(output, train_label)
                total_loss_train += train_loss.item()
                
                # Calculate the accuracy
                predicted_label = output.argmax(dim=1)
                acc = (predicted_label == train_label).sum().item()
                total_acc_train += acc
        # Update our model
                model.zero_grad()
                train_loss.backward()
                optimizer.step()
                step += 1 # Only for calculating dynamic accuracy
                
                # wandb.log({"training loss": train_loss.item(), "training accuracy": total_acc_train / (step*batch_size) })
            # ------ Evaluate model's performance using validation dataset -----------

            total_acc_val = 0
            total_loss_val = 0
            step = 0
      # torch.no_grad() will stop model from modifying parameters
            with torch.no_grad():

                for val_input, val_label, val_sample in tqdm(val_dataloader):

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)

                    val_loss = criterion(output, val_label)
                    total_loss_val += val_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                    step += 1
                    # wandb.log({"validation loss": val_loss.item(), "validation accuracy": total_acc_val / (step*batch_size)})
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')
    return model

def evaluate(model, tokenizer, test_data, method_name):
    test = Dataset(test_data, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=16)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    
    list_predicted_label = []
    total_acc_test = 0
    model.eval()
    with torch.no_grad():
        for test_input, test_label, test_sample in tqdm(test_dataloader):
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            predicted_label = output.argmax(dim=1)
            acc = (predicted_label == test_label).sum().item()
            list_predicted_label.extend(predicted_label.tolist())
            total_acc_test += acc
        list_predicted_label = ["Related" if x==1 else "Unrelated" for x in list_predicted_label]
        test_data[f"{method_name}_prediction"] = list_predicted_label
        
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return test_data


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



def main():
    # Suppose that input files all have annotations
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", help = "The path to the json file of training configurations", default="model_config.json")
    args = parser.parse_args()
    train_config = args.train_config

    with open(train_config, "r") as json_input:
        dict_train_config = json.load(json_input)
    EPOCHS, LR, batch_size, drop_out, input_file, model, save_model = (dict_train_config[key] for key in dict_train_config)

    torch.cuda.empty_cache() # This helps when the cuda memory is not enough
    model, tokenizer = load_models(model)
    model = My_Classifier(my_model = model, dropout = drop_out)
    df_train, df_val, df_test = load_dataset(input_file, 100)
    trained_model = train(tokenizer, model, df_train, df_val, LR, EPOCHS, batch_size)
    test_data = evaluate(trained_model, tokenizer, df_test, "Bert_ft")
    print(method_evaluation(test_data, "Bert_ft"))

main()