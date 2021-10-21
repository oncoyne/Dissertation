import torch
import numpy as np
import os
#import captum
import dask.dataframe as dd
import sklearn

#Uncomment these and above when using the interpret function

#from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
#from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation, DeepLiftShap, Lime, KernelShap

from model import TraderNet

from multiprocessing import cpu_count
from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch import nn, optim
from torch.nn import functional as F
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
#from torch_lr_finder import LRFinder

#Variables
learning_rate = 0.000015
run = 1
featureNumber = 13

#Create Directories
if not os.path.exists(Path("Saves")):
        os.mkdir(Path("Saves"))

if not os.path.exists(Path("Logs")):
        os.mkdir(Path("Logs"))

#---------------------------------------------------------------------------------------------------------------------------------------

#Helper Functions

#Cross-validation against the test set
def validate(model, val_set, device, val_split):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, prices in val_set:
            #Load up data
            batch = batch.to(device)
            prices = prices.to(device)
            forward_pass = model.forward(batch)
            loss = loss_fn(forward_pass, prices)
            total_loss += loss.item()
    
    average_loss= total_loss/val_split
    print("Average Loss: ", average_loss)
    model.train()

#Apply Captum Model interpretability methods to a trained network
def interpret(model, val_set, device):
    #torch.backends.cudnn.enabled=False
    #print(torch.load("save_epoch_9")["model"])
    model.load_state_dict(torch.load("GBM6_epoch_60")["model"])
    model = model.to(device)
    model.eval()

    feat, prices = next(iter(val_set))
    #Original Data
    #base = torch.FloatTensor([0.529372, 0.540780, 0.412285, 0.076678, 0.411272, 0.411267, 0.433313, 0.384094, 0.035099, 0.504892, 0.445274, 0.481291, 0.095117])
    #GBM Data
    base = torch.FloatTensor([0.520361, 0.528050, 0.173693, 0.059128, 0.173946, 0.173939, 0.175481, 0.179392, 0.013879, 0.507477, 0.457743, 0.154563, 0.109865])
    base = torch.unsqueeze(base, 0)
    #print(feat)
    #print(base)

    features = feat.to(device)
    baseline = base.to(device)

    USE_CUDA = True
    #basePrice = model.forward(baseline)
    #print(basePrice)
    #denormPrice = (basePrice * 174) + 49
    #print(denormPrice)

    
    #print("Features")
    #print(features)
    #print("Model Output")
    #print((model.forward(features) * 174) + 49)

    dl = GradientShap(model)
    dl_attr_test = dl.attribute(features, baseline)
    torch.save(dl_attr_test, "GBM_SHAP")

    #print("SHAP")
    #print(dl_attr_test)
    #print("Denorm")
    #finalDenorm = dl_attr_test.float() * 174
    #]print(finalDenorm)
    #dl_attr_test_sum = dl_attr_test.detach().cpu().numpy().sum(0)
    #dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

    
    #print(dl_attr_test_norm_sum)

#Save Model
def save_model(model, epoch: int, loss):
    location = str(Path("Saves"))+"/IPRZI_"+str(epoch)
    torch.save({'model': model.state_dict(), 'loss': loss.values, 'current_epoch': epoch}, location)

#---------------------------------------------------------------------------------------------------------------------------------------

#Data Ingress

#Set to use CUDA when on BC4, with a cpu as backup
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#Describes the structure of the dataset so that PyTorch can accurately load data
class BSEDATA(data.Dataset):
    def __init__(self, dataset_path):
        dataframe = dd.read_hdf(dataset_path,"/data")
        self.dataset = dataframe.values.compute()

    def __getitem__(self, ind):
        features = self.dataset[ind][:featureNumber]
        features = torch.from_numpy(features).float()
        output =  self.dataset[ind][featureNumber]
        output = torch.tensor(output).float()
        
        return features, output

    def __len__(self):
        return len(self.dataset)

#Input dataset
inputData = BSEDATA("GBM.hdf")

#Split into training and validation sets
train_split = round(0.9 * inputData.__len__())
val_split = round(0.1 * inputData.__len__())
train, val = torch.utils.data.random_split(inputData, [train_split, val_split])

#Used to split the validation set in half when applying Captum methods to it
#half = int(np.floor(len(val)/2))

#Load both the train and test set using PyTorch Dataloader
train_set = torch.utils.data.DataLoader(
        train,
        shuffle=True,
        batch_size= 16384,
        num_workers=cpu_count(),
        pin_memory=True
    )

val_set = torch.utils.data.DataLoader(
        val,
        shuffle=True,
        batch_size= 16384,
        num_workers=cpu_count(),
        pin_memory=True
    )

#---------------------------------------------------------------------------------------------------------------------------------------

#Network Setup

#Initialise the model class
model = TraderNet(number_features=featureNumber)


#Set loss function = MSE loss
loss_fn = nn.MSELoss()
#Set optimiser = ADAM
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Use this for captum stuff but you have to comment out everything below it
#interpret(model, val_set, device)

#Training variables (including Meades LR decay rate)
start_epoch = 0
end_epoch = 61
#lr_decay = 0.000001
step = 0

#Send Model to GPU/CPU
model = model.to(device)

#Minor experiment with a method called learning rate finder, didn't pay off
#lr_finder = LRFinder(model, optimiser, loss_fn, device="cuda")
#lr_finder.range_test(train_set, val_loader=val_set, end_lr=0.01, num_iter=100, step_mode="linear")
#lr_finder.plot(log_lr=False)

#Begin writing Logs + Print to output file
summary_writer = SummaryWriter(str(Path("Logs"))+"/IPRZI_"+str(run), flush_secs=5)
print("Learning Rate: "+str(learning_rate))
print("Beginning Training")

#---------------------------------------------------------------------------------------------------------------------------------------

#Trainining

for epoch in range(start_epoch, end_epoch):
    model.train()

    #Constant Learning Rate Decay (Meades)
    #templr = max((learning_rate - (epoch * lr_decay)),0)
    #print(templr)
    #for group in optimiser.param_groups:
       #group['lr'] = templr


    for batch, prices in train_set:
        #Load up data
        batch = batch.to(device)
        prices = prices.to(device)

        #Apply forward pass of the network
        forward_pass = model.forward(batch)
        #Calculate Loss   
        loss = loss_fn(forward_pass, prices)
        #Differentiate that loss
        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

        #Add data to logs
        summary_writer.add_scalars("loss", {"train": float(loss.item())}, step)
        step += 1
    
    
    print("Epoch: "+str(epoch)+ " complete")
    
    #Save and cross-validate the model every 5 epochs
    if ((epoch % 5)==0): 
        validate(model, val_set, device, val_split)
        save_model(model, epoch, loss)


summary_writer.close()

#validate(model, val_set, device, val_split)

print("Training Complete")


