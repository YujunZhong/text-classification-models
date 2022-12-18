from torch import nn


class TextClassificationModel(nn.Module):
    """ A model of a simple neural network."""
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        """ Weights initialization.
        :return: 
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        """ Model forward propagation
        :param text: the input text
        :param offsets: the offset is a tensor of delimiters to represent the beginning index of the individual sequence in the text tensor.
        :return: model outputs
        """
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)