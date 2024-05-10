"""
Define the observation model using 
1 linear regression
2 neural network
"""


import torch
import scipy.stats
import utils
from torch.utils.data import TensorDataset, DataLoader

## ---- HYPERPARAMETERS ---- ##
config_obsnet = dict(
    batch_size=64,
    hidden_size=64,
    learn_rate=0.001,
    epochs=1000,
    seq_len=25,
)

## ---- DATALOADING ---- ##
datafile = "regression_dataset.csv"  # sys.argv[1]
# create dataloaders from the datafile
tensors = utils.importSeqTensors(
    datafile, config_obsnet["seq_len"]
)  # use 70/10/20 split by default
train_loader = DataLoader(
    TensorDataset(tensors[2], tensors[3]), batch_size=config_obsnet["batch_size"]
)
test_loader = DataLoader(
    TensorDataset(tensors[4], tensors[5]), batch_size=config_obsnet["batch_size"]
)

y_train = tensors[2]  # input matrix of observations (train)
y_train = y_train.view(((y_train.shape[0] * y_train.shape[1]), y_train.shape[2]))

y_test = tensors[4]  # input matrix of observations (test)
y_test = y_test.view(((y_test.shape[0] * y_test.shape[1]), y_test.shape[2]))

x_train = tensors[3]  # the target variable
x_train = x_train.view(((x_train.shape[0] * x_train.shape[1]), x_train.shape[2]))
xtm_train = x_train[:-1, :]  # all of y except the last row
xt_train = x_train[1:, :]  # all of y except the first row

x_test = tensors[5]  # the target variable
x_test = x_test.view(((x_test.shape[0] * x_test.shape[1]), x_test.shape[2]))
xtm_test = x_test[:-1, :]  # all of y except the last row
xt_test = x_test[1:, :]  # all of y except the first row


###################################
### ---- LINEAR REGRESSION ---- ###
###################################
A = (xt_train.T @ xtm_train) @ torch.pinverse(
            xtm_train.T @ xtm_train
        )  # calculates kinematic state model

C = (y_train.T @ x_train) @ torch.pinverse(
            x_train.T @ x_train
        )  # calculates neural observation model


# Check A: MSE error and correlation between A*x_t and x_t+1
target = xt_test
pred = xtm_test @ (A.T)
MSE_error_A = ((target - pred) ** 2).mean()
corr_A, p_junk = scipy.stats.pearsonr(target.flatten(), pred.flatten())
print("MSE error for A_LR: ", MSE_error_A)
print("Correlation for A_LR: ", corr_A)

# Check C: MSE error and correlation between C*x_t and y_t
target = y_test
pred = x_test @ (C.T)
MSE_error_C = ((target - pred) ** 2).mean()
corr_C, p_junk = scipy.stats.pearsonr(target.flatten(), pred.flatten())
print("MSE error for C_LR: ", MSE_error_C)
print("Correlation for C_LR: ", corr_C)


################################
### ---- NEURAL NETWORK ---- ###
################################
# Define the neural network for observation model
class Obs_Net(torch.nn.Module):
    def __init__(self, in_shape, out_shape):
        super(Obs_Net, self).__init__()
        
        self.lstm = torch.nn.LSTM(in_shape, config_obsnet["hidden_size"], batch_first=True)
        self.fc = torch.nn.Linear(config_obsnet["hidden_size"], out_shape)

    def forward(self, x):
        # Check the dimensionality of the input tensor
        if x.ndim == 3:  # Batched input
            out, _ = self.lstm(x)
            y = self.fc(out)
        elif x.ndim == 2:  # Unbatched input
            x = x.unsqueeze(0)  # Add a batch dimension
            out, _ = self.lstm(x)
            y = self.fc(out).squeeze(0)  # Remove the batch dimension
        else:
            raise ValueError("Input tensor should have 2 or 3 dimensions.")
        return y
    
    # training the neural network
    def train_obs_nn(self, train_loader):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=config_obsnet["learn_rate"])
        for epoch in range(config_obsnet["epochs"]):
            for i, (y, x) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        # save the model
        torch.save(self.state_dict(), "obs_net_state_dict.pth")
    
    # testing the neural network
    def test_obs_nn(self, test_loader):
        self.eval()
        outputs = torch.zeros(config_obsnet["batch_size"], config_obsnet["seq_len"], y_train.shape[1])
        target_all = torch.zeros(config_obsnet["batch_size"], config_obsnet["seq_len"], y_train.shape[1])
        for i, (y, x) in enumerate(test_loader):
            outputs = torch.cat((outputs, self(x).squeeze(1)), 0)
            target_all = torch.cat((target_all, y), 0)
        # delete the init zeros, Merge the first two dims of outputs and target_all
        outputs = outputs[config_obsnet["batch_size"]:]
        target_all = target_all[config_obsnet["batch_size"]:]
        outputs = outputs.view(-1, y_train.shape[1])
        target_all = target_all.view(-1, y_train.shape[1])
        # Calculate the MSE error and correlation
        MSE_error_C = ((target_all - outputs) ** 2).mean()
        corr_C, p_junk = scipy.stats.pearsonr(target_all.detach().numpy().flatten(), outputs.detach().numpy().flatten())
        print("MSE error for C_NN: ", MSE_error_C)
        print("Correlation for C_NN: ", corr_C)


# Train and test the obs neural network
# obs_net = Obs_Net(x_train.shape[1], y_train.shape[1])
# # obs_net.train_obs_nn(train_loader)
# # load state dict
# obs_net.load_state_dict(torch.load("obs_net_state_dict.pth"))
# # test the model
# obs_net.test_obs_nn(train_loader)  
# # obs_net.test_obs_nn(test_loader)  


    