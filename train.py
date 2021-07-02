import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn
import torch_geometric.data
import time
from tqdm import tqdm
import graphnet
import argparse
from sklearn import neighbors
import pickle
from configobj import ConfigObj

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_training=None):
        """ 
        Dataset for the depth stratification
        """
        super(Dataset, self).__init__()

        # Read the training database
        with open('database/training_cmass.pk', 'rb') as filehandle:
            self.cmass_all = pickle.load(filehandle)

        with open('database/training_tau.pk', 'rb') as filehandle:
            self.tau_all = pickle.load(filehandle)

        with open('database/training_T.pk', 'rb') as filehandle:
            self.T_all = pickle.load(filehandle)            

        with open('database/training_logdeparture.pk', 'rb') as filehandle:
            self.dep_all = pickle.load(filehandle)
    
        # Now we need to define the graphs for each one of the computed models
        # The graph will connect all points at certain distance. We define this distance
        # as integer indices, so that we make sure that nodes are connected to the neighbors
        self.n_training = len(self.T_all)

        # Initialize the graph information
        self.edge_index = [None] * self.n_training
        self.nodes = [None] * self.n_training
        self.edges = [None] * self.n_training
        self.u = [None] * self.n_training
        self.target = [None] * self.n_training

        # Loop over all training examples
        for i in tqdm(range(self.n_training)):
            
            num_nodes = len(self.cmass_all)
            index_cmass = np.zeros((num_nodes, 1))
            
            index_cmass[:, 0] = np.arange(num_nodes) #self.cmass_all

            # Build the KDTree
            self.tree = neighbors.KDTree(index_cmass)

            # Get neighbors
            receivers_list = self.tree.query_radius(index_cmass, r=1)

            senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
            receivers = np.concatenate(receivers_list, axis=0)

            # Mask self edges
            mask = senders != receivers

            # Transform senders and receivers to tensors            
            senders = torch.tensor(senders[mask].astype('long'))
            receivers = torch.tensor(receivers[mask].astype('long'))

            # Define the graph for this model by using the sender/receiver information
            self.edge_index[i] = torch.cat([senders[None, :], receivers[None, :]], dim=0)

            n_edges = self.edge_index[i].shape[1]

            # Now define the nodes. For the moment we use only one quantity, the log10(T)
            self.nodes[i] = np.zeros((num_nodes, 1))
            self.nodes[i][:, 0] = np.log10(self.T_all[i])

            # We use to quantities for the information encoded on the edges: log(column mass) and log(tau)
            self.edges[i] = np.zeros((n_edges, 2))
            tau0 = np.log10(self.cmass_all[self.edge_index[i][0, :]])
            tau1 = np.log10(self.cmass_all[self.edge_index[i][1, :]])
            self.edges[i][:, 0] = (tau0 - tau1)

            tau0 = np.log10(self.tau_all[i][self.edge_index[i][0, :]])
            tau1 = np.log10(self.tau_all[i][self.edge_index[i][1, :]])
            self.edges[i][:, 1] = (tau0 - tau1)
                        
            # We don't use at the moment any global property of the graph, so we set it to zero.
            self.u[i] = np.zeros((1, 1))
            # self.u[i][0, :] = np.array([np.log10(self.eps_all[i][0, 0]), np.log10(self.ratio_all[i][0, 0])], dtype=np.float32)
            
            # We use the log10(departure coeff) as output, divided by 5 to make it closer to 1. In case a NaN is found, we
            # make them equal to zero
            self.target[i] = np.nan_to_num(self.dep_all[i][1][:, :].T / 5.0)           
            
            # Finally, all information is transformed to float32 tensors
            self.nodes[i] = torch.tensor(self.nodes[i].astype('float32'))
            self.edges[i] = torch.tensor(self.edges[i].astype('float32'))
            self.u[i] = torch.tensor(self.u[i].astype('float32'))
            self.target[i] = torch.tensor(self.target[i].astype('float32'))
        
    def __getitem__(self, index):

        # When we are asked to return the information of a graph, we encode
        # it in a Data class. Batches in graphs work slightly different than
        # in more classical situations. Since we have the connectivity of each
        # graph, batches are built by generating a big graph containing all
        # graphs of the batch.
        node = self.nodes[index]
        edge_attr = self.edges[index]
        target = self.target[index]
        u = self.u[index]
        edge_index = self.edge_index[index]

        data = torch_geometric.data.Data(x=node, edge_index=edge_index, edge_attr=edge_attr, y=target, u=u)
        
        return data

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        

class Formal(object):
    def __init__(self, configuration, batch_size=64, gpu=0, smooth=0.05, validation_split=0.2):
        
        # Is a GPU available?
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        
        # Factor to be used for smoothing the loss with an exponential window
        self.smooth = smooth        
        
        # If the nvidia_smi package is installed, then report some additional information
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size                

        
        kwargs = {'num_workers': 1, 'pin_memory': False} if self.cuda else {}        

        # Define the model. Read the configuration file and instantiate the model with the hyperparameters
        f = open(configuration, 'r')
        tmp = f.readlines()
        f.close()

        # Parse configuration file and transform to integers
        self.hyperparameters = ConfigObj(tmp)

        for k, q in self.hyperparameters.items():
            self.hyperparameters[k] = int(q)
        
        self.model = graphnet.EncodeProcessDecode(**self.hyperparameters).to(self.device)

        # Print the number of trainable parameters
        print('N. total trainable parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
    
        # Instantiate the dataset
        self.dataset = Dataset()

        # Randomly shuffle a vector with the indices to separate between training/validation datasets
        idx = np.arange(self.dataset.n_training)
        np.random.shuffle(idx)
        
        self.train_index = idx[0:int((1-validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-validation_split)*self.dataset.n_training):]

         # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)

        # Define the data loaders                
        self.train_loader = torch_geometric.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
        self.validation_loader = torch_geometric.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)

    def optimize(self, epochs, lr=3e-4):        

        best_loss = float('inf')

        self.lr = lr
        self.n_epochs = epochs        

        # Define the name of the model
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = f'weights/{current_time}.pth'

        print(' Model: {0}'.format(self.out_name))
        
        # Copy model
        shutil.copyfile(model.__file__, '{0}.model.py'.format(self.out_name))
        
        # Cosine annealing learning rate scheduler. This will reduce the learning rate with a cosing law
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Now start the training
        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):

            # Compute training and validation steps
            train_loss = self.train(epoch)
            valid_loss = self.validate()

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            # If the validation loss improves, save the model
            if  (valid_loss < best_loss):
                best_loss = valid_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'hyperparameters': self.hyperparameters,                    
                    'optimizer': self.optimizer.state_dict(),
                }
                
                print("Saving model...")
                torch.save(checkpoint, f'{self.out_name}')

            # Update the learning rate
            self.scheduler.step()

    def train(self, epoch):

        # Put the model in training mode
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        
        for batch_idx, (data) in enumerate(t):

            # Extract the node, edges, indices, target, global and batch information from the Data class            
            node = data.x
            edge_attr = data.edge_attr
            edge_index = data.edge_index
            target = data.y
            u = data.u
            batch = data.batch

            # Move them to the GPU
            node, edge_attr, edge_index = node.to(self.device), edge_attr.to(self.device), edge_index.to(self.device)
            u, batch, target = u.to(self.device), batch.to(self.device), target.to(self.device)
                        
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Evaluate Graphnet
            out = self.model(node, edge_attr, edge_index, u, batch)

            # Compute loss
            loss = self.loss_fn(out.squeeze(), target.squeeze())

            # Compute backpropagation
            loss.backward()
            
            # Update the parameters
            self.optimizer.step()
            
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            # Compute smoothed loss
            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            # Update information for this batch
            if (NVIDIA_SMI):
                usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=usage.gpu, memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)

        return loss_avg

    def validate(self):
        self.model.eval()
        loss_avg = 0
        t = tqdm(self.validation_loader)
        correct = 0
        total = 0
        n = 1
        with torch.no_grad():
            for batch_idx, (data) in enumerate(t):
                                
                node = data.x
                edge_attr = data.edge_attr
                edge_index = data.edge_index
                target = data.y
                u = data.u
                batch = data.batch

                node, edge_attr, edge_index = node.to(self.device), edge_attr.to(self.device), edge_index.to(self.device)
                u, batch, target = u.to(self.device), batch.to(self.device), target.to(self.device)
                
                out = self.model(node, edge_attr, edge_index, u, batch)

                loss = self.loss_fn(out.squeeze(), target.squeeze())
                
                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                                
                t.set_postfix(loss=loss_avg)            
   
        return loss_avg

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=100, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--batch', '--batch', default=128, type=int,
                    metavar='BATCH', help='Batch size')
    parser.add_argument('--split', '--split', default=0.08, type=float,
                    metavar='SPLIT', help='Validation split')
    parser.add_argument('--conf', '--conf', default='conf.dat', type=str,
                    metavar='CONF', help='Configuration file')
    
    
    parsed = vars(parser.parse_args())

    network = Formal(
            configuration=parsed['conf'],
            batch_size=parsed['batch'], 
            gpu=parsed['gpu'], 
            validation_split=parsed['split'], 
            smooth=parsed['smooth'])

    network.optimize(parsed['epochs'], lr=parsed['lr'])