# DL Samples for Text

Common method of `Natural Language Processing (NLP)` with these samples:

- TorchText package Usage

    - `TextCalssification.py` : Classify different text type via `AG_NEWS` provided by TorchText

- RNN & LSTM & GRU

    - `ClassifyNames.py` : Classify different country names via RNN without connvenience function of torchtext, so also introduce some basic operate of NPL.

    - `GeneratNames.py` : Generate different country names

    - `TranslationS2S.py` : Translate sequence to sequence

- Transformer *( based on paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) )*
    
    - `TransformerS2S.py` : Introduce how to train a sequence-to-sequence model that uses the `nn.Transformer` module. Just using transform encoden layers, with directly compare with output sequence embedded.

---

## Some Refer Construction image

#### The construction of RNN to classify names

![Image of RNN](https://i.imgur.com/Z2xbySO.png)

#### The construction of RNN to generate names

![Image of RNN](https://i.imgur.com/jzVrf7f.png)

#### The translation with a sequence to sequence Network and Attention

- attention mechanism

    ![Image of Attention mechanism](https://pytorch.org/tutorials/_images/seq2seq.png)

- the seq2seq models

  - Encoder 
  
    ![Image of encoder](https://pytorch.org/tutorials/_images/encoder-network.png)

  - Decoder
  
    ![Image of decoder](https://pytorch.org/tutorials/_images/decoder-network.png)

  - AttentionDecoder

    ![Image of Attention decoder](https://pytorch.org/tutorials/_images/attention-decoder-network.png)

#### The Architecture of Transformer

![transformer architecture](https://pytorch.org/tutorials/_images/transformer_architecture.jpg)
