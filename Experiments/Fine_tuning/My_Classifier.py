import torch
class My_Classifier(torch.nn.Module):
    def __init__(self, my_model, dropout=0.5):
        super(My_Classifier, self).__init__()
        self.model = my_model
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 2)
        # self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, input_id, mask):
        output = self.model(input_ids= input_id, attention_mask=mask)
        last_hidden = output.last_hidden_state
        # pooler_output = torch.mean(last_hidden, dim=1)
        pooler_output = output.pooler_output # [CLS] classification
        dropout_output = self.dropout(pooler_output)
        linear_output = self.linear(dropout_output)
        # final_layer = self.relu(linear_output)
        final_layer = self.softmax(linear_output)
        return final_layer
