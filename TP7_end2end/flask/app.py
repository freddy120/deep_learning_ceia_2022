from flask import Flask

import pandas as pd
import numpy as np
import torch

app = Flask(__name__)


class NNetWithEmbeddings(torch.nn.Module):

  def __init__(self, number_of_vendors, embedding_dim):
    super().__init__()
    self.embedding = torch.nn.Embedding(num_embeddings=number_of_vendors, embedding_dim=embedding_dim)
    self.linear_1 = torch.nn.Linear(in_features=(13 + embedding_dim), out_features=200, bias=True)
    self.relu_1 = torch.nn.ReLU()
    self.linear_2 = torch.nn.Linear(in_features=200, out_features=100, bias=True)
    self.relu_2 = torch.nn.ReLU()
    self.linear_3 = torch.nn.Linear(in_features=100, out_features=1, bias=True)

  def forward(self, x, vendor_idx):
    # dim x = batch_size, 13
    # dim vendor_idx = 1024, 1
    vendor_emb = self.embedding(vendor_idx) #  1024, embedding_dim
    final_input = torch.cat([x, vendor_emb], dim=1) # 1024, embedding_dim + 13
    x = self.linear_1(final_input)
    x = self.relu_1(x)
    x = self.linear_2(x)
    x = self.relu_2(x)
    x = self.linear_3(x)
    return x


model = NNetWithEmbeddings(500, 8)
model.load_state_dict(torch.load("static/modelWithEmb.torch", map_location=torch.device('cpu')))
model.eval()


@app.route('/')
def hello_world():  # put application's code here
    x = torch.tensor(np.array([0.2479, 0.0200, 0.2530, 0.0077, 0.0548, 0.1736, 0.2903, 0.0671, 0.3566,
        0.2921, 0.4783, 0.0000, 1.0000])).float()
    vendor_idx = torch.tensor(np.array([326])).int()

    print(x.shape)
    print(x.reshape(1, -1).shape)
    prediction = model(x.reshape(1, -1), vendor_idx)

    return "hello " + str(torch.sigmoid(prediction).item())


if __name__ == '__main__':
    app.run()
