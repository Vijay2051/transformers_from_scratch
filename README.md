# transformers_from_scratch
implementing the transformers paper from scratch.


# https://github.com/pbloem/former
Code is made up of 9 files for a total of 500 LOC
Good habit is setup a ```environment.yml``` file https://github.com/pbloem/former/environment.yml

1. We want to learn  a weight from each input state to each output state
2. Each weight represents the correlation of each input state to each other output state
3. We want the weight to sum upto 1

After we come with weights we can just do a forward or backward propwithout thinking 
about it too much

Attention is something you can do without learning the weights, but you can learn them 
from the data in self supervised way like the next sentence prediction

Query: compute meanig vector for your own output
key: compute the relationship between the word input_i and word output_j
Value: compute the relationship between the word_i and all other outputs

We are gonna learn a new matrix for each Q, K, V
and then we gonna take a dot product of all the three and then do a softmax

Since the embedding matrix are huge in dimension, softmax on this thing
will cause a huge loss in gradients and so we'll be dividing them by the
```sqrt of the (embedding-dimension)```

We do this operation for different values of Q, K, V and then we append all the representations.
This is Multi-head attention, Really dont know how this gonna help, but it is a trent in ML that 
aggregation really helps!!

There's two ways to do self attention
* keep full size for each head
* scale head size

It also depends on how much memory and compite power you have

Its not obvious how to apply batch norm to sequences so we apply a
layer normalisation which will normalises all the neurons

## code walkthrough
/former

Modules
* Wide attention
* Narrow attention
* transformer block

Transformer Module
* Generating transformer
* Classification Transformer

Classification Transformers
* setup data using torchtext
* setup model from the cli using the model parameters
* make prediction, look at the each actual label and do a backward prop in loss
* no_grad measure accuracy


