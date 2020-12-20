import torch
import torch.nn as nn
import numpy as np 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#getting the data
data = open('file_name.txt', 'r').read() 
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("no of data = " ,data_size)
print("no of unique chars = " ,vocab_size)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

# converting data to one hot:

data_idx = np.array([char_to_ix[f] for f in data])
shape = (data_idx.size, data_idx.max()+1)
data_one_hot = np.zeros(shape)
rows = np.arange(data_idx.size)
data_one_hot[rows, data_idx] = 1

data_one_hot = np.transpose(data_one_hot)

## h_t  = W_h.h_t-1 + W_x.x_t + b
# so we are going to concatenate the output with the input
'''
here I'm actually not going to follow the tutorial
simple nothing fancy just one hidden state

'''

class rnn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(rnn, self).__init__()
        self.hidden_size = hidden_size

        self.hidden = nn.Linear(hidden_size  + input_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, prev_hidden):

        hidden = self.hidden(prev_hidden)
        output = np.max(self.softmax(hidden))
        return hidden , output 

    def initHidden(self): 
        return torch.zeros(1, self.hidden_size)

learning_rate = 0.0005

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)


def train(train_data, hidden_size):
    
    h_0 = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    prev_hidden = torch.zeros(1, hidden_size)
    x = 0
    for i in train_data:
        
        while i != train_data[-1]:

            output, hidden = rnn( prev_hidden)
            prev_hidden = hidden
            l = criterion(output, train_data[x+1])
            x+=1
            loss += 1

    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("train_loss:",loss)



    

        