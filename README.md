# Chinese poetry geneator
--A digital poet using basic RNN models

It's a prictice when learning RNN and it's vaiants such as GRU, LSTM(with and without peephole).
It's based on a project provided by my Python teacher. The project, which gives me necessary data to bulid and test a model, was written by
rainym00d, Github: https://github.com/rainym00d and 
Ethan00Si, Github: https://github.com/Ethan00Si

I trained an LSTM(just one layer) for 420 epochs, and it performed quite well, yet the .pth file is more 100mb so I was unable to push it onto Github.The last thing I did was add LSTM with peephole and using amp and tensorboard provided by Pytorch, yet I haven't try these.

It's such a basic model and have a lot of room for improvement. If you have any suggestions, please contact me so that I can learn from you and make it better.

If you're also learning dl or think it's interesting, please give a star and here is my next project - a English to Spanish translator using Encoder-Decoder and Attention model, at https://github.com/Ariza-Ding/English-Spanish-translator

# How to use

Clone or download the code and set the model you want to use in main.py, if you want to use the model provided in the models folder, you should change their name to model.pth or change the script to find these files. Just as their, it's the parameter for the basical RNN model.

As I mentioned above, I also trained LSTM for 420 epochs, if you want that parameters, just contact me.
