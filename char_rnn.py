import torch
import torch.nn as nn
import numpy as np 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#getting the data
data = open('txt', 'r').read() 
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("no of data = " ,data_size)
print("no of unique chars = " ,vocab_size)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

# converting data to one hot tensor:

data_idx = np.array([char_to_ix[f] for f in data])
shape = (data_idx.size, data_idx.max()+1)
data_one_hot = np.zeros(shape)
rows = np.arange(data_idx.size)
data_one_hot[rows, data_idx] = 1

data_one_hot = torch.tensor(data_one_hot, dtype=torch.float) 

## h_t  = W_h.h_t-1 + W_x.x_t + b
# so we are going to concatenate the output with the input
'''
here I'm not going to follow the tutorial
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
        return hidden 


learning_rate = 0.0005

model = rnn(input_size=85, hidden_size=85)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


def train(train_data, hidden_size, EPOCHS ):


    model.zero_grad()

    loss = 0

    prev_hidden = torch.zeros(1, hidden_size)


    for j in range(EPOCHS):

        l =0 

        for i in train_data:
            
            x = 0

            while x < train_data.shape[1]-1:
                
                hidden = model( prev_hidden)
                prev_hidden = hidden
                b = train_data[x+1]
                b = b.unsqueeze(0)

                l += criterion(hidden, b)
                x+=1

        print("train_loss for epoch", j,"is:" ,l)

        #backprop
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

def test():
    with torch.no_grad:
        model()