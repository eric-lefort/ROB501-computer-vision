import torch
import torch.utils.data 
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### set a random seed for reproducibility (do not change this)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

### Set if you wish to use cuda or not
use_cuda_if_available = True
print(f"Using CUDA: {use_cuda_if_available and torch.cuda.is_available()}")

### Define the Convolutional Neural Network Model
class CNN(torch.nn.Module):
    def __init__(self, num_bins): 
        super(CNN, self).__init__()
        
        # dropout doesn't seem to help
        # batch normalization helps a lot

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.0)
        self.dropout2d = torch.nn.Dropout2d(0.0)

        # conv2d
        self.conv1 = torch.nn.Conv2d(3, 16, stride=1, kernel_size=(5, 5))
        self.pool1 = torch.nn.MaxPool2d((2,2),stride=2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        
        # conv2d
        self.conv2 = torch.nn.Conv2d(16,32, stride=1, kernel_size=(5,5))
        self.pool2 = torch.nn.MaxPool2d((2,2),stride=2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        
        # conv2d
        self.conv3 = torch.nn.Conv2d(32, 64, stride=(1,2), kernel_size=(3, 5))
        self.bn3 = torch.nn.BatchNorm2d(64)

        # conv2d
        self.conv4 = torch.nn.Conv2d(64, 128, stride=(1,2), kernel_size=(5, 5))
        self.bn4 = torch.nn.BatchNorm2d(128)
        
        # conv2d
        self.conv5 = torch.nn.Conv2d(128, 256, stride=1, kernel_size=(2,4))
        # self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=(3, 5))

        # dense layer
        self.dense1 = torch.nn.Linear(256, num_bins)

        if use_cuda_if_available and torch.cuda.is_available():
            self = self.cuda()

    ###Define what the forward pass through the network is
    def forward(self, x):
        
        # conv2d
        x = self.conv1(x)
        x = self.dropout2d(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        # conv2d
        x = self.conv2(x)
        x = self.dropout2d(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.bn2(x)
        
        # conv2d
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.pool2(x)
        
        # conv2d
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)

        # conv2d
        x = self.conv5(x)
        x = self.relu(x)
        x = x.squeeze() # (Batch_size x num_bins x 1 x 1) to (Batch_size x num_bins)
        x = self.dropout(x)

        # dense layer
        x = self.dense1(x)

        return x

### Define the custom PyTorch dataloader for this assignment
class dataloader(torch.utils.data.Dataset):
    """Loads the KITTI Odometry Benchmark Dataset"""
    def __init__(self, matfile, binsize=45, mode='train'):
        self.data = sio.loadmat(matfile)
        
        self.images = self.data['images']
        self.mode = mode

        # Fill in this function if you wish to normalize the input
        # Data to zero mean.
        self.normalize_to_zero_mean()
        
        if self.mode != 'test':

            # Generate targets for images by 'digitizing' each azimuth 
            # angle into the appropriate bin (from 0 to num_bins)
            self.azimuth = self.data['azimuth']
            bin_edges = np.arange(-180,180+1,binsize)
            self.targets = (np.digitize(self.azimuth,bin_edges) -1).reshape((-1))

    def normalize_to_zero_mean(self):
        self.images -= np.mean(self.images)

    def __len__(self):
        return int(self.images.shape[0])
  
    def __getitem__(self, idx):
        if self.mode != 'test':
            return self.images[idx], self.targets[idx]    
        else:
            return self.images[idx]

if __name__ == "__main__": 
    '''
    Initialize the Network
    '''
    binsize=20
    bin_edges = np.arange(-180,180+1,binsize)
    num_bins = bin_edges.shape[0] - 1
    cnn = CNN(num_bins) #Initialize our CNN Class
    
    '''
    Uncomment section to get a summary of the network (requires torchsummary to be installed):
        to install: pip install torchsummary
    '''
    from torchsummary import summary
    inputs = torch.zeros((1,3,68,224))
    summary(cnn, input_size=(3, 68, 224))
    
    '''
    Training procedure
    '''
    
    CE_loss = torch.nn.CrossEntropyLoss(reduction='sum') #initialize our loss (specifying that the output as a sum of all sample losses)
    params = list(cnn.parameters())
    optimizer = torch.optim.Adam(params, lr=40e-4, weight_decay=0.0) #initialize our optimizer (Adam, an alternative to stochastic gradient descent)
    
    ### Initialize our dataloader for the training and validation set (specifying minibatch size of 128)
    dsets = {x: dataloader(f'{x}.mat', binsize=binsize) for x in ['train', 'val']}
    # dsets = {x: dataloader('sun-cnn_{}.mat'.format(x),binsize=binsize) for x in ['train', 'val']} 
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=128, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    #### TESTING
    # # Combine datasets
    # from torch.utils.data import ConcatDataset

    # # dsets = {x: dataloader(f'{x}.mat', binsize=binsize) for x in ['train', 'val']}
    # combined_dataset = ConcatDataset([dsets['train'], dsets['val']])
    # combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=128, shuffle=True, num_workers=4)
    # dsets['train'] = combined_dataset
    # dset_loaders['train'] = combined_dataloader

    
    loss = {'train': [], 'val': []}
    top1err = {'train': [], 'val': []}
    top5err = {'train': [], 'val': []}
    best_err = 1
    
    ### Iterate through the data for the desired number of epochs
    for epoch in range(0,25):
        for mode in ['train', 'val']:    #iterate 
            epoch_loss=0
            top1_incorrect = 0
            top5_incorrect = 0
            if mode == 'train':
                cnn.train(True)    # Set model to training mode
            else:
                cnn.train(False)    # Set model to Evaluation mode
                cnn.eval()
            
            dset_size = dset_loaders[mode].dataset.__len__()
            for image, target in dset_loaders[mode]:    #Iterate through all data (each iteration loads a minibatch)
                
                # Cast to types and Load GPU if desired and available
                if use_cuda_if_available and torch.cuda.is_available():
                    image = image.cuda().type(torch.cuda.FloatTensor)
                    target = target.cuda().type(torch.cuda.LongTensor)
                else:
                    image = image.type(torch.FloatTensor)
                    target = target.type(torch.LongTensor)

                optimizer.zero_grad()    #zero the gradients of the cnn weights prior to backprop
                pred = cnn(image)   # Forward pass through the network
                minibatch_loss = CE_loss(pred, target)  #Compute the minibatch loss
                epoch_loss += minibatch_loss.item() #Add minibatch loss to the epoch loss 
                
                if mode == 'train': #only backprop through training loss and not validation loss       
                    minibatch_loss.backward()
                    optimizer.step()        
                        
                
                _, predicted = torch.max(pred.data, 1) #from the network output, get the class prediction
                top1_incorrect += (predicted != target).sum().item() #compute the Top 1 error rate
                
                top5_val, top5_idx = torch.topk(pred.data,5,dim=1)
                top5_incorrect += ((top5_idx != target.view((-1,1))).sum(dim=1) == 5).sum().item() #compute the top5 error rate
    
                
            loss[mode].append(epoch_loss/dset_size)
            top1err[mode].append(top1_incorrect/dset_size)
            top5err[mode].append(top5_incorrect/dset_size)
    
            print("{} Loss: {}".format(mode, loss[mode][epoch]))
            print("{} Top 1 Error: {}".format(mode, top1err[mode][epoch]))    
            print("{} Top 5 Error: {}".format(mode, top5err[mode][epoch])) 
            if mode == 'val':
                print("Completed Epoch {}".format(epoch))
                if top1err['val'][epoch] < best_err:
                    best_err = top1err['val'][epoch]
                    best_epoch = epoch
                    torch.save(cnn.state_dict(), 'best_model_{}.pth'.format(binsize))
                
           
    print("Training Complete")
    print("Lowest validation set error of {} at epoch {}".format(np.round(best_err,2), best_epoch))        
    '''
    Plotting
    '''        
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.grid()
    ax1.plot(loss['train'],linewidth=2)
    ax1.plot(loss['val'],linewidth=2)
    #ax1.legend(['Train', 'Val'],fontsize=12)
    ax1.legend(['Train', 'Val'])
    ax1.set_title('Objective', fontsize=18, color='black')
    ax1.set_xlabel('Epoch', fontsize=12)
    
    ax2.grid()
    ax2.plot(top1err['train'],linewidth=2)
    ax2.plot(top1err['val'],linewidth=2)
    ax2.legend(['Train', 'Val'])
    ax2.set_title('Top 1 Error', fontsize=18, color='black')
    ax2.set_xlabel('Epoch', fontsize=12)
    
    ax3.grid()
    ax3.plot(top5err['train'],linewidth=2)
    ax3.plot(top5err['val'],linewidth=2)
    ax3.legend(['Train', 'Val'])
    ax3.set_title('Top 5 Error', fontsize=18, color='black')
    ax3.set_xlabel('Epoch', fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig('net-train.pdf')
