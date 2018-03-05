# TextClassifier

This is a text-classifier using nerual networks ,the data is book reviews which are obtained by my teammates using web crawler.

At first, I used the pretrained embeddings which are download from a CSDN blog .The blogger says he using 120G corpus to train the 60W+ chinese word vectors by word2vec.But I can't say that those are good.But I try to use my own corpus to get the word2vec model and then I observed that the accuracy improved at least 1%.So I think that perhaps the word vector is too sensitived to its field.

By now,I have three indepentent model:TextCNN,TextRNN and bi-GRU with attention.

The excel file "parameters recording" is my recording for adjusting parameters."bi-gru_2" is saying that the RNN cell is 2 forward gru cell and 2 backward gru cell.The number of epoch is the epoch when the test accuracy get the maximum value and after that the accuracy is becoming less(overfitting).Most recordings is trained with the 120G corpus word2vec model.Though the test accuracy is so better with the new word2vec model,the relative value is still meaningfull.

In my experiment，the CNN model performs poor.I think the reason is the length of sentences has a serious differentiation:there are lost of sentences of 1 or 2 words and sentences of 10+ words. 3 layers LSTMs model and bi-GRU with attention model get the best and closely performance,with the 1.0 dropout keep probability.LSTM model is best suitable to l2_loss rate in 10e-6,and the attention model is best suitable to 10e-5.Dropout always has bad impact CNN and attention models ,and seems to be unrelated to the performance of LSTMs model(make training slowly of course).I want to know why.

Limited by my GPU,I'm sorry to I could only use the adam optimizer for efficiency.
