# time-series-attribution

1. Please download uv, covid, and mobility in form of .pkl at https://drive.google.com/drive/folders/1oG-ePVmhf0oTOrArlwvF3R6bqy0oECd7?usp=sharing

2. Run python unify.py to extract merged uv, covid, and mobility features into "dfunity.pkl"

3. Run python unify_test.py to extract merged uv, covid, and mobility features into "dfunity_test.pkl"

4. Run deep-lstm-autoencoder.py to train conv-lstm

5. Run deep-lstm-autoencoder-val.py to test trained conv-lstm

6. Run grad-cam-lstm.py to obtain visual attribution to time-series prediction

7. Run test_matplotlib.py to plot the visual attribution


Reference:

Yudistira, N., Sumitro, S. B., Nahas, A., & Riama, N. F. (2021). Learning where to look for COVID-19 growth: Multivariate analysis of COVID-19 cases over time using explainable convolution-LSTM. Applied Soft Computing, 107469.
