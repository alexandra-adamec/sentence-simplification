# Sentence simplification

**Deep learning** based sentence simplification solved as a **machine translation** task (translate a complex sentence into a simple sentence). 

## Simplification model architecture

This project includes the implementations of three RNN simplification models. They all follow the **encoder-decoder architecture**, successfully used in neural machine translation. Two of the models use simple RNNs in the encoder with **GRU** resp. **LSTU** units. The third one uses a **bidirectional LSTM RNN** with **attentions** (alignments). The experiments showed the latter is more successful in simplification of long sentences compared to the others.

### Encoder-decoder model
An encoder-decoder model is a combination of two RNNs mapping a variable-length source sequence to its related variable-length target sequence. The encoder reads sequentially words from a source sentence creating its fixed length **vector representation**. The decoder **sequentially generates** the words of the target sentence in the context of the encoded source sentence. In fact, the decoder is an **RNN language model** of the target language. The main benefit of the encoder-decoder is the **joint architecture** and **training** of the translation and language model.

The proposed **RNN encoder-decoder** for the **sentence simplification**:
 
![RNN encoder-decoder model](/images/rnn-simpl-model.png)

The model computes the probability *y<sub>i</sub>* of the *i<sup>th</sup>* word of the target sentence *y<sub>1:L</sub>* being simplified from the input complex sentence *x<sub>1:M</sub>* (where L, M are lengths of simplified and original sentences respectively).

<img src="https://latex.codecogs.com/svg.latex?p(y_i|y_{1:i-1},x_{1:M})"/> 

*y<sub>i</sub>* has a form of the **probability distribution** over the **simple language vocabulary**. The probability of the entire output sentence *y<sub>1:L</sub>* is a **product of the probabilities** of each word.

<img src="https://latex.codecogs.com/svg.latex?p(y_{1:L}|x_{1:M})=\prod_{i=1}^{L}p(y_i|y_{1:i-1},x_{1:M})"/>

The neural network consists of several connected layers.

#### Projection layers
The role of the projection layer is to project words into the **continuous vector space** (word embeddings). A similar projection layer P is in the encoder and decoder (assuming that the simple vocabulary is a subset of the complex vocabulary).

#### Hidden layers
The following layer in the encoder is the **recurrent hidden layer**. It serves as a **memory unit** remembering the sentence history. At each time step *j*, a new state of the memory *h<sub>j</sub>* is computed as a linear combination of the new word *x<sub>j</sub>* with the previous state *h<sub>j-1</sub>*. This combination is transformed by a **non-linear activation unit** (in this project we experimented with GRU, LSTM respectively).

<img src="https://latex.codecogs.com/svg.latex?h_j=f(x_j,h_{j-1})=f(Wx_j+Uh_{j-1}+b_h)"/>

(*b<sub>h</sub>* is the bias term)

The last hidden state forms the output from the encoder and it represents a **fixed-length encoding** of the entire **input sentence**. This is later used to feed the decoder with the **context information** *c = h<sub>M</sub>*.

Similarly, the decoder also consists of the **recurrent hidden layer**, remembering the **history of the target words** that have been already **generated**. A new state *s<sub>i</sub>* is computed from the previous state *s<sub>i-1</sub>*, the last word of the target sequence *y<sub>i-1</sub>* and the encoded input sentence *c*. A non-linear activation function *g(y)* is similar to the hidden layer function in the encoder.

<img src="https://latex.codecogs.com/svg.latex?s_i=g(y_{i-1},s_{i-1},c)=g(W'y_{i-1}+U's_{i-1}+Cc+b_s)"/>

#### Output layer

The encoder-decoder outputs a prediction of the next target word following the sequence of already generated words in the given context. The prediction depends on the current hidden state *s<sub>i</sub>*, a previously generated word *y<sub>i_1</sub>* and the encoded input sentence *c*. A **softmax function** helps to produce a **valid probability distribution** at the output layer, with all the values being positive and summing to one.

<img src="https://latex.codecogs.com/svg.latex?y_i=softmax(s_i,y_{i-1},c)=softmax(Vs_{i}+Dy_{i-1}+Ec+b_y)"/>

It is very difficult to train a basic encoder-decoder model. In particular, it is challenging to learn **long-span dependencies** between the two mapped sequences. A trick that might help is to **revert the order of words** in the input sentences (in both training and test data sets). With this approach, it is easier to learn the dependencies between the beginnings of the two mapped sentences, which increases the chances of success at this region and adds to the overall accuracy.


### Bi-directional LSTM encoder-decoder
Bi-directional RNN models enable to scan the variable-length input sentences in **both directions in parallel**. It equals the chances to learn the dependencies from the beginning and from the end of the sentences. This model is based on the work of *Bahdanau et al.*: 
> who used a bi-direction RNN encoder for the machine translation. Their model combines not only the language and the translation model, but adds also the alignment model mapping the related portions of the input and output sentences (currently known as attentions mechanism). Such model was also able to translate long sentences with a higher accuracy.
> 
> --- Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

The final sentence simplification model differs in the type of the activation units used in the hidden layers. Instead of the GRU it uses the LSTM.

![Bi-directional RNN encoder-decoder model](/images/bi-rnn-simpl-model.png)

#### Bi-directional encoder
The model consists of a bi-directional encoder. The **forward encoder** reads a sentence in the direction from the beginning to the end. The **backward encoder** uses the opposite direction, from the end of the sentence to the beginning. Both return a sequence of the created hidden states, which are **concatenated** and grouped by the time steps. The alignment model of the decoder uses a sequence of these concatenated hidden states.

#### Alignment model (attentions)
Compared to the previous models, each generated target word *y<sub>i</sub>* depends on a specific context *c<sub>i</sub>*. *c<sub>i</sub>* encodes a particular portion of the input sentence related to the target word *y<sub>i</sub>* by a **soft alignment**. It is computed as a weighted sum of the encoder’s sequence of concatenated hidden states parameterized by the *i<sup>th</sup>* row of an **alignment matrix** *A*. *A<sub>i,j</sub>* expresses the probability of the *i<sup>th</sup>* target simple word being related to the *j<sup>th</sup>* complex word and its surrounding.


<img src="https://latex.codecogs.com/svg.latex?s_i=g(y_{i-1},s_{i-1},c_i)=g(W'y_{i-1}+U's_{i-1}+Cc_i+b_s)"/>

<img src="https://latex.codecogs.com/svg.latex?c_i=A_i[h_{forward}h_{backward}]"/>

A row *A<sub>i</sub>* has a size of the input sequence and it is computed with a simple **feed-forward neural network** playing a role of the **alignment model**. The alignment model, as a part of the decoder, is **trained jointly** with the translation and language model. The input to the alignment network is the previous hidden state of the decoder *s<sub>i-1</sub>* and the entire sequence of the encoder’s hidden states [h<sub>forward</sub> h<sub>backward</sub>]. They are linearly combined into *E<sub>i</sub>* consequently transformed by the softmax function to the **probability distribution** of the *i<sup>th<sup>* **word alignments** over the input words.



