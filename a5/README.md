# Assignment 5: Written Part

## 1. Character-based convolutional encoder for NMT

- (a): Yes, CNN can operate over variable lengths regardless of the input size, which is also an important advantage of FCN in vision community.
- (b): We should use default 1 padding on both sides to match the conv filter kernel size
- (c): Residual connections work very well in deep CNN because it can at least representing an identity mapping and the highway connections can provide effective gradient flow. It's better to initialize bias  term as zero since bias term represents a prior distribution and when it equals 0, the highway module equals an identity mapping. 
- (d): better gradient flow and more parallel-friendly calculation



## 2. Character-based LSTM decoder for NMT

- (e): After 10 hours‘ training, we get the final BLEU score: 36.28.



## 3. Analyzing NMT Systems

- (a): traducir, traduzco, traduce occurs but traduces, traduzca, traduzcas. They have similar word meaning but just are used in different grammar situations, which is unaware for word-level NMT. However, character-based NMT has the ability to learn this.

  