# SCAT

SCAT: A Time Series Forecasting with Spectral Central Alternating Transformers

SCAT design consists of two critical components: (i) employing each spectral clustering center of time series as the focal point for attention computation; (ii) utilizing alternating attention, where each query is transformed by a token that is interoperable with spectral clustering centers, and the attention execution dimension is performed at the sequence level instead of the token level. Our Spectral Central Alternating Transformer significantly improves the univariate prediction input accuracy of multivariate input compared to the state-of-the-art method (SOTA) in time series forecasting, especially in power fields.

## SCAT vs. Transformers & Linear

**1. Series Clustering Center**

SCAT addresses limitations in data acquisition equipment by employing a preprocessing step that involves clustering and classifying the meteorological time series data.  This process identifies and separates central points within each sub-sequence, each representing distinct weather patterns. Subsequently, SCAT utilize these spectral clustering centers as computational cores to eliminate the redundancy of traditional attention computing structures and effectively capture global semantic information, enhancing the overall predictive performance of our model.

**2. Alternating  Attention**

Unlike the prevalent transformer-based models that perform feature extraction based on query-key connections among tokens, alternating attention utilizes clustering centers as attention computation cores. This unique approach considers both Euclidean distance and feature distance between tokens. After nonlinear propagation through activation functions, SCAT implement sequence-level attention filtering instead of token-level attention, thereby preserving the global characteristics of tokens within each channel to the maximum extent.

![1704708101251](https://github.com/Nickname1233/SCAT/assets/155961174/6c53c4c9-74e0-4c43-82f6-07a6fbf3733b)

![d62d091b0bc9d318bd42198c2a170ee](https://github.com/Nickname1233/SCAT/assets/155961174/208de6a5-ed21-4078-8545-03614899f667)

![34485a8c33ea04f8e2a1eea15bcd2f7](https://github.com/Nickname1233/SCAT/assets/155961174/a26ddf63-fa29-4e30-b933-69ef7cbf1883)


## Get Started

1. Install Python 3.6, PyTorch 1.7.1.
2. Download data. You can obtain all the six benchmarks from `./data`. The datasets password can be obtained by contacting the author. **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:


## Baselines

We will keep adding series forecasting models to expand this repo:

- [x] Autoformer
- [x] FEDformer
- [x] TimesNet
- [x] PatchTST
- [x] DLinear
- [x] HI
- [x] STID
- [x] FEDformer
- [x] Reformer
- [x] ETSformer
- [x] MICN
- [ ] N-BEATS
