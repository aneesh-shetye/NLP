In this repo I have implemented some common Natural Language Processing models for sequence modelling and neural machine translation. 

1. Charachter level RNN from scratch: 
I have implemented charachter level RNN for generating new dinosaur name. I have used PyTorch. 
However I have not used any of its predefined RNN functions and made RNN from scratch. 

2. Charachter level RNN using predefined RNN, LSTM and GRU functions: 
I have implemented the same above model but instead of building the RNN architecture from scratch I have used the RNN, LSTM and GRU functions from PyTorch. 

3. Sequence modelling: 
I have made a sequence model that generates sentences in the same style of the sentences the model was trained on. I have used Charles Dicken's " A Christmas Ghost" to train the model. 
I have used multilayered LSTM blocks for the same. 

4. Neural Machine Translation using Seq-2-Seq: 
Here I have made a English to French translating model. It uses the classical Encoder-Decoder architecture.
I have used Multi-30k dataset for training and testing of the model.

5. Neural Machine Translation using Attention: 
This is the implementation of the attention paper. Here the encoder hidden states are values and the encoder hidden state at the current time step is the query. 
I have implemented two kinds of attention here. Additive and dot product based and have used multilayered GRU blocks for both encoder and decoder. 
The encoder has bidirectional RNN's as specified in the paper. 


