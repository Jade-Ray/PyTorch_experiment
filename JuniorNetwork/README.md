# Some Junior DL-Networks of PyTorch

Junior networks worked simple & shallow Task with these samples:

- Basical Operation with PyTroch network

    - `quick_build_net.py` : Creating Net inherited `torch.nn.Module` to DL

    - `NeuralNetworks.py` : Creating complexity Net includes convolutional layer to DL

    - `optimizer_speed_up_train.py` : Showing different `Optimizer` effection of Net training

    - `batch_train.py` : Using mutil-batch to train Net

- Linear Regression

    - `linear_regression.py` : Only via one `nn.Linear` to train regression qustion with `MSE`

- Logical Regression

    - `area_classification.py` : Only via one `nn.Linear` to train regression qustion with `CrossEntropyLoss`

- Auto Encoder & Decoder

    - `autoencoder_minst.py` : Encoding `MINST Dataset` to three features and decoding to origin image

- CNN

    - `cnn_classifier_minst.py` : Using Convolutional Neural Network to train classifier of `MINST Dataset`

    - `cnn_classifier_cifar10.py` : Using Convolutional Neural Network to train classifier of `CIFAR10 Dataset`

- RNN

    - `rnn_classifier_minst.py` : Using Recurrent Neural Network to train classifier of `MINST Dataset`

    - `rnn_regressor_sin2cos.py` : Using Recurrent Neural Network to train linear regression of `sin` to `cos`