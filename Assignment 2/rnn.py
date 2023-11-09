import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import mlflow

unk = '<UNK>'


# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='relu')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        _, hidden = self.rnn(inputs)
        # [to fill] obtain output layer representations
        output = self.W(hidden)
        # [to fill] sum over output 
        summed_output = torch.sum(output, dim=0)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(summed_output)
        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument("--model_path", type=str, required=False, default="best_model.pth",
                        help="path to save the trained model")  # UPDATED: added argument for model path
    args = parser.parse_args()

    # UPDATED: Only run the training block if do_train is True
    if args.do_train:
        print("========== Loading data ==========")
        train_data, valid_data = load_data(args.train_data,
                                           args.val_data)  # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

        # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
        # Further, think about where the vectors will come from. There are 3 reasonable choices:
        # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
        # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
        # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
        # Option 3 will be the most time consuming, so we do not recommend starting with this
        mlflow.start_run()

        print("========== Vectorizing data ==========")
        model = RNN(50, args.hidden_dim)  # Fill in parameters
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

        mlflow.log_param("hidden dimension", args.hidden_dim)
        mlflow.log_param("Optimizer", "Adam_0.0005")
        stopping_condition = False
        epoch = 0

        last_train_accuracy = 0
        last_validation_accuracy = 0
        count = 0

        best_validation_accuracy = 0  # keep track of the best validation accuracy
        best_model = None  # to keep a copy of best model

        while not stopping_condition:
            random.shuffle(train_data)
            model.train()
            # You will need further code to operationalize training, ffnn.py may be helpful
            print("Training started for epoch {}".format(epoch + 1))
            train_data = train_data
            correct = 0
            total = 0
            minibatch_size = 8
            N = len(train_data)

            loss_total = 0
            loss_count = 0
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    input_words = " ".join(input_words)

                    # Remove punctuation
                    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                    # Look up word embedding dictionary
                    vectors = [
                        word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                        in input_words]

                    # Transform the input into required shape
                    vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                    output = model(vectors)

                    # Get loss
                    example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))

                    # Get predicted label
                    predicted_label = torch.argmax(output)

                    correct += int(predicted_label == gold_label)
                    # print(predicted_label, gold_label)
                    total += 1
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss

                loss = loss / minibatch_size
                loss_total += loss.data
                loss_count += 1
                loss.backward()
                optimizer.step()
            print(loss_total / loss_count)
            mlflow.log_metrics(({"epoch": epoch, "train_loss": (loss_total / loss_count).item()}))
            print("Training completed for epoch {}".format(epoch + 1))
            print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            mlflow.log_metrics({"epoch": epoch, "train_acc": correct / total})
            trainning_accuracy = correct / total

            model.eval()
            correct = 0
            total = 0
            random.shuffle(valid_data)
            print("Validation started for epoch {}".format(epoch + 1))
            valid_data = valid_data

            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk']
                           for i
                           in input_words]

                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                # print(predicted_label, gold_label)
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            validation_accuracy = correct / total
            mlflow.log_metric("valid_acc", validation_accuracy)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_model = model.state_dict()
                model_directory = os.path.dirname(args.model_path)

                model_save_path = args.model_path  # Use the path provided, or default if not provided
                model_directory = os.path.dirname(model_save_path)

                if model_directory and not os.path.exists(model_directory):
                    os.makedirs(model_directory, exist_ok=True)

                torch.save(best_model, model_save_path)
                print(f"Model saved to {model_save_path} with validation accuracy: {best_validation_accuracy}")

            if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
                stopping_condition = True
                print("Training done to avoid overfitting!")
                print("Best validation accuracy is:", last_validation_accuracy)
            else:
                last_validation_accuracy = validation_accuracy
                last_train_accuracy = trainning_accuracy

            epoch += 1
            if (epoch > args.epochs):
                stopping_condition = True

    if args.test_data != "to fill" and not args.do_train:
        print("========== Testing ==========")
        test_data, _ = load_data(args.test_data, args.val_data)  # Assuming test data is loaded the same way
        model = RNN(50, args.hidden_dim)
        word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))
        # Load the model from the path if provided
        if args.model_path and os.path.isfile(args.model_path):

            model.load_state_dict(torch.load(args.model_path))
            model.eval()

            correct = 0
            total = 0
            print("Testing ")

            for input_words, gold_label in tqdm(test_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk']
                           for i
                           in input_words]

                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                if (predicted_label != gold_label):
                    print("Input words: ", input_words)
                    print("predicted label: ", predicted_label.item())
                    print("Actual/gold label: ", gold_label)
                correct += int(predicted_label == gold_label)
                total += 1
                # print(predicted_label, gold_label)
            print("Test accuracy : {}".format(correct / total))
