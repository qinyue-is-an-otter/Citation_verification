# Requirements: Need to import torch and numpy
import torch
import numpy as np
labels={
    "Related" : 1,
    "Unrelated" : 0,
}

# Redefine Dataset class to split data and use batch more easily
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df,tokenizer):
        self.labels = [labels[label] for label in df["Label"]]
        self.texts = [tokenizer(df['Citation_context'][index], df['Cited_content'][index],
                                padding='max_length', 
                                max_length = 512, 
                                truncation=True,
                                return_tensors="pt") 
                      for index in df.index]
        self.samples = [(df['Citation_context'][index], df['Cited_content'][index]) for index in df.index]

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    
    def get_batch_samples(self, idx):
        return self.samples[idx]

    # This is the most important one, you need to return the texts (matrix) and its labels to fit into the expected output
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_labels = self.get_batch_labels(idx)
        batch_samples = self.get_batch_samples(idx)
        return batch_texts, batch_labels, batch_samples