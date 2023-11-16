# SongGeneration-LSTM

The provided code implements a text generation model using a recurrent neural network (RNN) with LSTM (Long Short-Term Memory) cells. The text generation is based on an input dataset containing song lyrics by the artist Drake. Below is the methodology for the code:

*Data Loading and Cleaning:

The code begins by loading a dataset (drake_data.csv) containing song lyrics into a Pandas DataFrame. It cleans the lyrics by removing square brackets and their contents, replacing slashes with spaces, and splitting the lyrics into separate lines.

*Tokenization and Text Preprocessing:

The cleaned lyrics are converted to lowercase and stored in the corpus variable. The Tokenizer class from Keras is used to tokenize the text into sequences of integers, where each word is represented by a unique integer.

*N-gram Sequence Generation:

The n_gram_seqs function generates a list of n-gram sequences from the tokenized corpus. It considers sequences of increasing lengths, capturing the context for predicting the next word.

*Padding Sequences:

The pad_seqs function pads the generated n-gram sequences to have the same length. This is necessary for creating consistent input sequences for the neural network.

*Feature-Label Splitting:

The features_and_labels function separates the n-gram sequences into features (input) and labels (output). Labels are one-hot encoded to match the vocabulary size.

*Model Architecture:

The create_model function defines a Sequential model using Keras. It includes an Embedding layer to represent words in a continuous vector space, a Bidirectional LSTM layer for capturing contextual information bidirectionally, and a Dense layer with softmax activation for predicting the next word.

*Model Training:

The untrained model is created using the defined architecture. The model is trained on the features and labels using the categorical crossentropy loss and the Adam optimizer.

*Training Visualization:

The code uses Matplotlib to visualize the training process by plotting training accuracy and loss over epochs.

*Text Generation:

The final part of the code generates new text given a seed text using the trained model. The seed text is tokenized, padded, and iteratively expanded by predicting the next word based on the model's output probabilities.

*Output Display:

The generated text is printed, providing an example of the model's ability to generate coherent sequences based on the learned patterns from the input lyrics.

It's important to note that the effectiveness of the text generation heavily depends on the size and diversity of the training dataset, as well as the chosen model architecture and hyperparameters. Adjustments to these factors may be necessary for optimal performance. Additionally, the code could benefit from comments to enhance readability and understanding.
