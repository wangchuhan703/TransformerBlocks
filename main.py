import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import TransformerEncoder
from transformer import FeedForwardClassifier

from utilities import Utilities

from transformer import TransformerDecoder
import torch.optim as optim

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer    2e-4 1e-4
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())

    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer

    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels

def compute_classifier_accuracy(encoder, classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    encoder.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device).long(), Y.to(device)
            embeddings, attn_maps = encoder(X)
            outputs = classifier(embeddings.float())
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        encoder.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity




def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def main():
    # Part 1
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

    # Initialize encoder, classifier, and optimizer
    encoder = TransformerEncoder(n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, vocab_size=tokenizer.vocab_size).to(device)
    classifier = FeedForwardClassifier(n_input=n_input, n_hidden=n_hidden, n_output=n_output).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)

    # Training loop for the classifier
    for epoch in range(epochs_CLS):
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass
            embeddings, attn_maps = encoder(xb)   # embeddings.shape == (batch_size, n_embd)
            predictions = classifier(embeddings)    # predictions.shape == (batch_size, n_output)
            loss = torch.nn.functional.cross_entropy(predictions, yb)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute accuracy after each epoch
        accuracy = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
        print(f"Epoch {epoch+1}/{epochs_CLS}, Loss: {loss.item()}, Accuracy: {accuracy}%")

    # Part 1.4 -- Sanity Check
    encoder.eval()
    utils = Utilities(tokenizer, encoder)
    test_sentence = "But I stand before you tonight because all across America something is stirring."
    # test_sentence = "I will rebuild our military to meet future conflicts."
    utils.sanity_check(test_sentence, block_size=block_size)

    # Part 1.5 -- decoder parameters number
    total_params = sum(p.numel() for p in encoder.parameters())

    print(f"Total number of parameters in encoder: {total_params}")


    # Part 2
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()

    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    # Initialize decoder model and optimizer for language modeling
    decoder = TransformerDecoder(n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size,
                                 vocab_size=tokenizer.vocab_size).to(device)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate)

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)

        # LM training code here
        optimizer_decoder.zero_grad()
        loss = decoder(xb, yb)  # decoder computes cross-entropy loss
        loss.backward()
        optimizer_decoder.step()

        # Evaluation for perplexity at intervals
        if i % eval_interval == 0:
            perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
            print(f"Iteration {i}, Loss: {loss.item()}, Perplexity: {perplexity}")

    # Evaluation on test data
    print("Evaluating on test sets...")
    test_files = ["speechesdataset/test_LM_obama.txt", "speechesdataset/test_LM_wbush.txt", "speechesdataset/test_LM_hbush.txt"]
    for test_file in test_files:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_text = f.read()
        test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        perplexity = compute_perplexity(decoder, test_loader, eval_iters)
        print(f"Test Perplexity for {test_file}: {perplexity}")

    # Part 2.3 -- Sanity Check
    decoder.eval()
    utils = Utilities(tokenizer, decoder)

    # 测试解码器的注意力机制
    test_sentence = "We cannot put our faith in the word of tyrants, who solemnly sign non-proliferation treaties, and then systemically break them."
    utils.sanity_check(test_sentence, block_size=block_size, is_decoder=True)

    # Part 2.4 -- decoder parameters number
    total_params = sum(p.numel() for p in decoder.parameters())

    print(f"Total number of parameters in decoder: {total_params}")



if __name__ == "__main__":
    main()
