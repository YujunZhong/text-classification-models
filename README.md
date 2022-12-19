# Text Classification Models
This repository contains code for text multi-classification tasks. The following four models are implemented:
- Naive Bayes classifier using Bag of words feature
- Linear SVM
- Neural network
- Long short-term memory network (LSTM)

## Run command
### Naive Bayes and SVM
#### Before training naive bayes and SVM models, we need to generate new dataset files:
<code>
cd tools

python data_process.py
</code>

#### Train the Naive bayes model:
<code>python nb_svm_train.py nb</code>

#### Train the SVM model:
<code>python nb_svm_train.py svm</code>

#### Test a Naive bayes or SVM model:
<code>python nb_svm_inference.py svm</code>

### SVM with word2Vec
Pay attention to the data path. Follow the steps in the 'svm_with_word2vec.ipynb'.

### Neural nets
#### Train a simple neural network:
<code>python nn_train.py</code>

#### Test a simple neural network:
<code>python nn_inference.py</code>

### LSTM and Explainability of LSTM
Pay attention to the data path. Follow the steps in the 'lstm.ipynb'.
