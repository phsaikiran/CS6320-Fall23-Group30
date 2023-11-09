# NLP Assiggment 2

One example on running the code:

**FFNN**

``python ffnn.py --hidden_dim 16 --epochs 5 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json --test_data ./Data_Embedding/test.json``

**RNN**

``python rnn.py --hidden_dim <hidden_dimensions> --epochs 25 --train_data <train_data-path> --val_data <valid_data-path> --do_train --model_path <path to save model>``

To test:

``python rnn.py --hidden_dim 32 --epochs 5 --train_data training.json --val_data validation.json --test_data test.json --model_path <trained mode path>``

- The hidden dimension must be same for traning and testing.
- model_path argument is optional, there is default filename mentioned.
