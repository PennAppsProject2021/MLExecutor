import argparse
import ExperimentModel from UserModel
import hyperparameters
import Data
from opacus import PrivacyEngine

# Instantiate the parser
parser = argparse.ArgumentParser(description='Train your model in a way that is differentially private')

parser.add_argument('exp_name', type=string, help='The name of your experiment (required)')

args = parser.parse_args()

exp_name = args.exp_name

learning_rate = hyperparameters.getLearningRate()
training_iters = hyperparameters.getTrainingIters()
report_iter = hyperparameters.getReportIter()

model = UserModel()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.learning_rate)

privacy_engine = PrivacyEngine(
    model,
    sample_rate=0.01,
    alphas=[1, 10, 100],
    noise_multiplier=1.3,
    max_grad_norm=1.0,
)
privacy_engine.attach(optimizer)

losses = []
domain = Data.getDomain()

for i in range(training_iters):
    x_batch, y_batch = Data.getBatch()

    preds = model(x_batch)
    loss = loss_fn(preds, y_batch)
    loss.backward()
    optimizer.step()

    if i % report_iter == 0:
        plt.figure(i+10)
        plt.plot(x, model(domain))
        plt.savefig(f"{exp_name}/Animation.png")

        plt.figure(i+10.1)
        plt.plot(np.linspace(0, len(losses), len(losses)), losses)
        plt.savefig(f"{exp_name}/TrainLoss.png")

#save this model for later use
torch.save(model.state_dict(), f"{exp_name}/{exp_name}.model")
