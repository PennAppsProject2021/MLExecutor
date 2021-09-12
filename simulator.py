import torch
import numpy as np
import matplotlib.pyplot as plt
from opacus import PrivacyEngine

class Unit(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(Unit, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class ExampleModel(torch.nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        hidden_dim = 40
        num_hidden_layers = 20
        self.first = torch.nn.Linear(1, hidden_dim)
        self.hidden = torch.nn.Sequential(*[Unit(hidden_dim)])
        self.last = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.first(x)
        x = self.hidden(x)
        x = self.last(x)
        return x




total_pts = 50
x = np.linspace(-5, 5, total_pts)
y = x ** 2
y[total_pts // 2] = 50
#y[(total_pts // 2) + 1] = 10

"""
plt.figure(0)
plt.plot(x, y)
plt.show()
"""

num_training_iters = 50000
batch_size = 20

model = ExampleModel()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

"""
Differential Privacy hear we go?

privacy_engine = PrivacyEngine(
    model,
    sample_rate=0.01,
    alphas=[1, 10, 100],
    noise_multiplier=1.3,
    max_grad_norm=1.0,
)
privacy_engine.attach(optimizer)
#that's it (apparently)
"""





losses = []

report_iter = 200

for i in range(num_training_iters):
    indices = np.random.randint(total_pts, size=batch_size)
    x_batch = x[indices]
    y_batch = y[indices]
    """
    plt.figure(i+1)
    plt.scatter(x_batch, y_batch)
    plt.show()
    """
    x_batch = torch.tensor(x_batch)

    x_batch = x_batch.unsqueeze(dim=1).float()
    #print(x_batch.shape)
    #go()
    y_batch = torch.tensor(y_batch).unsqueeze(dim=1).float()


    preds = model(x_batch)
    loss = loss_fn(preds, y_batch)
    loss.backward()
    optimizer.step()
    a = loss.item()
    losses.append(a)
    print(a)

    if i % report_iter == 0:
        plt.figure(i+10)
        plt.plot(x, model(torch.tensor(x).unsqueeze(dim=1).float()).detach().numpy())
        plt.title("Not Differentially Private!")
        plt.savefig("animation.png")





plt.figure(3.1)
plt.plot(np.linspace(0, len(losses), len(losses)), losses)
plt.show()

plt.figure(4.1)
plt.plot(x, model(torch.tensor(x).unsqueeze(dim=1).float()).detach().numpy())
plt.show()
