# Assignment 4: Written Part

## 1. Neural Machine Translation with RNNs

- (g): We use `enc_masks`  to filter out attention score belonging to `<PAD>` because we don't want to calculate our posterior probability distribution according to padding.
- (i): Corpus BLEU: 35.9207748118086
- (j): 
  - dot product attention to multiplicative attention:
    - advantage: easy and efficient
    - disadvantage: limited expressive ability and the dimension of encoder hidden state and decoder hidden state should always be the same
  - additive attention to multiplicative attention
    - advantage: additive attention works like a simple FC neuron network so should be more expressive
    - disadvantage: more computational time and more data demand