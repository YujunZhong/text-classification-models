from torch import nn
import torch.nn.functional as F
import torch


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)



# class TextClassificationModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_class):
#         super(TextClassificationModel, self).__init__()
        
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(embed_dim, 256, (k, embed_dim)) for k in (1)])
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(256 * 3, num_class)

#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x

#     def forward(self, text, offsets):
#         out = self.embedding(text, offsets)
#         out = out.unsqueeze(1)
#         out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out