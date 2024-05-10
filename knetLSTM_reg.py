import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import utils
import sys
from model import Obs_Net
import scipy.stats
# import wandb

# 1. Train nonlinear observation model and compare it to the linear observation model. Train them independently of everything else.
# 2. Then, do PCA on the observations, and run the same experiment as 1.
# 3. Then, take the best result, and compare training the observation model end to end, or independently
# 4. (Optional) Try a non linear compression model and test everything else with it

## ---- TAKE IN DATAFILE (and other stuff?) AS COMMAND LINE ARGS ---- ##
datafile = "regression_dataset.csv"  # sys.argv[1]

## ---- HYPERPARAMETERS ---- ##
config = dict(
    batch_size=64,
    hidden_size=48,
    learn_rate=0.001,
    epochs=20,
    seq_len=25,
    NN_obs_model=True,
    load_trained_obs=True,
)

## ---- DATALOADING ---- ##
# create dataloaders from the datafile
# TODO: you could apply the transformation (PCA or whatever) when you load the tensors
tensors = utils.importSeqTensors(
    datafile, config["seq_len"]
)  # use 70/10/20 split by default
train_loader = DataLoader(
    TensorDataset(tensors[0], tensors[1]), batch_size=config["batch_size"]
)


## ---- BUILD KALMAN NET ---- ##
class KalmanNet(nn.Module):
    # initialize class members
    def __init__(self, hid_size, in_shape, out_shape, NN_obs_model=False, load_trained_obs=False, append_ones_y=True, device="cpu"):
        # initialize additional KF variables
        self.A, self.C, self.W, self.Q = (
            None,
            None,
            None,
            None,
        )  # state model, obs model, state noise, obs noise
        self.At, self.Ct = None, None  # useful transposes
        self.prev_pred = np.zeros((1, out_shape))  # initial previous prediction is 0
        self.m = 0
        self.n = 0
        self.NN_obs_model = NN_obs_model
        self.load_trained_obs = load_trained_obs
        self.append_ones_y = append_ones_y
        self.device = device
        # shapes of input and output (kinda like the nn dimensions, but the whole model)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hid_size = hid_size

        # initialize neural network
        super(KalmanNet, self).__init__()
        self.output_dim = self.in_shape * self.out_shape
        # two fully connected linear layers
        self.lstm = nn.LSTM(in_shape, self.hid_size, batch_first=True)  # input layer
        self.fc1 = nn.Linear(self.hid_size, self.output_dim)  # output layer

        hid_shape_obs = 64
        self.lstm_obs = nn.LSTM(out_shape, hid_shape_obs, batch_first=True)
        self.fc_obs = nn.Linear(hid_shape_obs, in_shape)
        

        if self.NN_obs_model:
            if self.load_trained_obs:
                # Load the saved checkpoint for the Obs_Net model
                self.obs_model = Obs_Net(out_shape, in_shape)
                self.obs_model.load_state_dict(torch.load('obs_net_state_dict.pth'))
            else:
                # Initialize a new instance of the Obs_Net model
                self.obs_model = Obs_Net(out_shape, in_shape)
        else:
            self.obs_model = None
        
    # train kf parameters
    def train_kf(
        self, x, y
    ):  # x is input matrix of observations, y is input matrix of ground truth state
        if self.append_ones_y:
            y.unsqueeze(-1)  # pads y with an extra column of ones if needed
        x = x.view(((x.shape[0] * x.shape[1]), x.shape[2]))
        y = y.view(((y.shape[0] * y.shape[1]), y.shape[2]))
        ytm = y[:-1, :]  # all of y except the last row
        yt = y[1:, :]  # all of y except the first row
        
        ### ---- State Model ---- ###
        self.A = (yt.T @ ytm) @ torch.pinverse(
            ytm.T @ ytm
        )  # calculates kinematic state model
        self.W = (
            (yt - (ytm @ self.A.T)).T @ (yt - (ytm @ self.A.T)) / (yt.shape[0] - 1)
        )  # covariance/noise for state model
        self.At = self.A.T
        self.m = self.A.size()[0]

        ### ---- Observation Model ---- ###
        # TODO: if using PCA, you need to convert x using the PCA transformation
        self.C = (x.T @ y) @ torch.pinverse(
            y.T @ y
        )  # calculates neural observation model
        self.Ct = self.C.T
        self.n = self.n = self.C.size()[0]
        if self.NN_obs_model:
            if not self.load_trained_obs:
                # train the observation model
                self.obs_model.train_obs_nn(train_loader)
                # set to eval mode
                self.obs_model.eval()
            else:
                # set to eval mode
                self.obs_model.eval()        
            self.Q = (
                (x - self.obs_model(y)).T @ (x - self.obs_model(y)) / yt.shape[0]
            )  # covariance/noise for obs model
        else:
            self.Q = (
                (x - (y @ self.C.T)).T @ (x - (y @ self.C.T)) / yt.shape[0]
            )  # covariance/noise for obs model
            

        # check the fits of everything
        # Compute correlation between self.A @ ytm and yt (corr_all function)
        target = yt
        pred = ytm @ (self.A.T)
        corr_A, p_junk = scipy.stats.pearsonr(target.flatten(), pred.flatten())
        print("Correlation for A_LR on train set: ", corr_A)
        # Compute correlation between self.C @ y and x (corr_all function)
        target = x
        if self.NN_obs_model:
            pred = self.obs_model(y).detach().numpy()
            corr_C, p_junk = scipy.stats.pearsonr(target.flatten(), pred.flatten())
            print("Correlation for C_NN on train set: ", corr_C)
        else:
            pred = y @ (self.C.T)
            corr_C, p_junk = scipy.stats.pearsonr(target.flatten(), pred.flatten())
            print("Correlation for C_LR on train set: ", corr_C)

    # one forward pass through kalman filter -- x is measurement, kg is kalman gain
    def kf_forward(self, x, kg):
        pred = torch.zeros(
            kg.shape[0], kg.shape[1], self.out_shape
        )  # size of prediction tensor: (batches, seq_len, params)
        for batch in range(kg.shape[0]):
            # TODO: previous prediction set to 0 at top of each sequence (?)
            self.prev_pred.fill(0)
            for t in range(kg.shape[1]):
                kg_tp = kg[batch, t, :].view(
                    (self.out_shape, self.in_shape)
                )  # reshape kalman gain for that data in batch
                # compute prior
                prev_pred_tens = (
                    torch.from_numpy(self.prev_pred).float() @ self.A
                )  # loss is worse but corr and mse better
                prev_pred_tens = prev_pred_tens.view((1, self.out_shape))
                if self.NN_obs_model:
                    pred[batch, t, :] = (
                        prev_pred_tens
                        + (((x[batch, t, :].unsqueeze(1)).T - self.obs_model(prev_pred_tens))
                           @ (kg_tp.T)) 
                    )
                else:
                    pred[batch, t, :] = (
                        prev_pred_tens
                        + (
                            kg_tp
                            @ (x[batch, t, :].unsqueeze(1) - self.C @ prev_pred_tens.T)
                        ).T
                    )  # state prediction
                self.prev_pred = (
                    pred[batch, t, :].detach().numpy()
                )  # updates internal storage of previous prediction
        return pred

    # one forward pass through neural network -- x is measurement
    def nn_forward_batch(self, x):
        # shape of x: (batches, seq_len, data, 1)
        h0 = torch.zeros(
            1, x.shape[0], self.hid_size
        )  # hidden layer shape: (1, data, hidden_size)
        c0 = torch.zeros(1, x.shape[0], self.hid_size)
        out, (h0, c0) = self.lstm(x, (h0, c0))
        out = self.fc1(out)
        return out

    # one epoch of nn training
    def train_nn(self, train_loader, optimizer, loss_fn):
        # model training
        train_loss = 0.0  # initialize loss at 0
        self.train()  # put network in training mode
        for (
            data,
            target,
        ) in (
            train_loader
        ):  # each batch should be 4D: (batches, num_seq, seq_length, data)
            # data = data.unsqueeze(-1) # slap a 1 on the end
            optimizer.zero_grad()  # clear error gradients
            out = self.nn_forward_batch(data)  # forward pass - output is kalman gain
            pred = self.kf_forward(
                data, out
            )  # compute prediction based on outputed kalman gain
            batch_loss = loss_fn(
                pred, target
            )  # compute loss between position output and target
            batch_loss.backward()  # backward pass the loss
            optimizer.step()  # optimize
            train_loss += batch_loss.item() * data.size(0)  # update loss for the batch
            # look more into why batch loss is updated exactly like this

        # normalize running loss over data size
        train_loss = train_loss / len(train_loader.dataset)

        # print training loss for each epoch
        print(f"Average training loss (MSE): {train_loss}")
        # wandb.log({"train_loss": train_loss})

    # validation done as one big pass
    def validate_nn(self, inputs, targets, loss_fn):
        valid_loss = 0.0
        self.eval()

        kg = self.nn_forward_batch(inputs)
        pred = self.kf_forward(inputs, kg)
        valid_loss = loss_fn(pred, targets)
        print(f"Validation loss (MSE): {valid_loss}")
        # wandb.log({"valid_loss": valid_loss})

    def model_train(
        self,
        train_loader,
        valid_inputs,
        valid_targs,
        epochs,
        optimizer,
        loss_fn=nn.MSELoss(),
    ):
        for e in tqdm(range(epochs)):
            print(f"Epoch : {e+1}/{epochs}")
            self.train_nn(train_loader, optimizer, loss_fn)
            self.validate_nn(valid_inputs, valid_targs, loss_fn)
        print("Done!")

    def predict(self, inputs):
        self.eval()
        kg = self.nn_forward_batch(inputs)
        pred = self.kf_forward(inputs, kg)
        return pred


## ---- DECLARE AND TRAIN KALMAN NET ---- ##
# with wandb.init(project="knetLSTM", config=config):
# declare model
kn = KalmanNet(config["hidden_size"], tensors[0].shape[-1], tensors[1].shape[-1], NN_obs_model=config["NN_obs_model"], load_trained_obs=config["load_trained_obs"])
optim = torch.optim.Adam(kn.parameters(), lr=config["learn_rate"])
# train kf parameters
kn.train_kf(tensors[0], tensors[1])
# train nn parameters
kn.model_train(train_loader, tensors[2], tensors[3], config["epochs"], optim)

## ---- TESTING AND ANALYSIS ---- ##
tpreds = kn.predict(tensors[4])
ttargs = tensors[5]

# reformatting prediction and target tensors to fit analysis functions
flat_targs = torch.reshape(
    ttargs, (ttargs.shape[0] * ttargs.shape[1], ttargs.shape[2])
)  # flatten to 2D
flat_preds = torch.reshape(tpreds, (tpreds.shape[0] * tpreds.shape[1], tpreds.shape[2]))
targ_params = torch.split(
    flat_targs, 1, dim=1
)  # split into a list of tensors, one tensor for each parameter of motion
pred_params = torch.split(flat_preds, 1, dim=1)

# for i in range(200):
#     wandb.log({"x pos": flat_preds[i, 0]})
#     wandb.log({"y pos": flat_preds[i, 1]})
#     wandb.log({"x vel": flat_preds[i, 2]})
#     wandb.log({"y vel": flat_preds[i, 3]})

# analysis functions
labels = ["X Position", "Y Position", "X Velocity", "Y Velocity"]
# graph ground truth vs. prediction
utils.truthPlot(
    targ_params,
    pred_params,
    labels,
    pts_graphed=200,
    title="Ground Truth vs. Prediction for Kalman LSTM",
)
mse = utils.mse_all(targ_params, pred_params, numpy=True)  # MSE
print("MSE error for KalmanNet on test set: ", mse)
corr = utils.corr_all(targ_params, pred_params)  # correlation
print("Correlation for KalmanNet on test set: ", corr)

# barplots of mse and corr
utils.barPlot(mse, labels, title="MSE for Kalman LSTM")
utils.barPlot(corr, labels, title="Correlation Coeficient for Kalman LSTM")
