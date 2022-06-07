import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def h_out_calculator(H_in, padding, dilation, kernel_size, stride):
    """
    Computes H_out from a Conv2d layer in pytorch
    """
    H_out = np.floor(
        (H_in + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    return H_out

def w_out_calculator(W_in, padding, dilation, kernel_size, stride):
    """
    Computes W_out from a Conv2d layer in pytorch
    """
    W_out = np.floor(
        (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1 ) / stride[1] + 1
    )
    return W_out

def findConv2dOutShape(H_in, W_in, conv, pool=2):
    """
    Reference:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    H_out = h_out_calculator(H_in, padding, dilation, kernel_size, stride)
    W_out = w_out_calculator(W_in, padding, dilation, kernel_size, stride)

    return H_out, W_out


class NaiveNet(nn.Module):
    """
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
                in_channels = 3,
                out_channels = 6,
                kernel_size = (3,3)
                )
        self.conv2 = nn.Conv2d(
                in_channels = 6,
                out_channels = 12,
                kernel_size = (3,3)
                )
        self.conv3 = nn.Conv2d(
                in_channels = 12,
                out_channels = 24,
                kernel_size = (3,3)
                )
        self.conv4 = nn.Conv2d(
                in_channels = 24,
                out_channels = 48,
                kernel_size = (3,3)
                )
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(43200, 4320)
        self.fc2 = nn.Linear(4320, 432)
        self.fc3 = nn.Linear(432, 40)
        self.fc4 = nn.Linear(40, 2)

    def forward(self, x):
        """
        The shape after running 
        input through all the 
        convolution and pooling
        layers is 
        43200
        I computed this manually 
        and am adding it manually
        this is not good practice
        but this is the best 
        I can do right now
        """
        x = F.relu(self.conv1(x))
        #print('Shape after first convolution: {}'.format(x.shape))
        x = self.pool(x)
        #print('Shape after first pool: {}'.format(x.shape))
        x = F.relu(self.conv2(x))
        #print('Shape after second convolution: {}'.format(x.shape))
        x = self.pool(x)
        #print('Shape after second pool: {}'.format(x.shape))
        x = F.relu(self.conv3(x))
        #print('Shape after third convolution: {}'.format(x.shape))
        x = self.pool(x)
        #print('Shape after third pool: {}'.format(x.shape))
        x = F.relu(self.conv4(x))
        #print('Shape after first convolution: {}'.format(x.shape))
        x = self.pool(x)
        #print('Shape after fourth pool: {}'.format(x.shape))
        x = torch.flatten(x, 1)
        #print('shape after flattening: {}'.format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, .2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, .2)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, .2)
        x = self.fc4(x)

        
        return x

def get_validation_loss(model,loss_fn, validation_dataloader):
    """
    Be careful with what your reduction on the 
    dataloader is.
    I am going to use 'mean'
    as the reduction and get the mean
    of the mean loss across all 
    minibatches in the training dataloader
    """
    running_loss = 0
    model.eval()

    for xb, yb in validation_dataloader:
        outputs = model(xb)
        running_loss += loss_fn(outputs, yb.to(torch.int64)).item()

    avg_loss = running_loss/ float(len(validation_dataloader))
    return avg_loss

def train_on_minibatch(model, inputs, labels, loss_fn, optimizer):
    """
    """
    model.train()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels.to(torch.int64))
    loss.backward()
    optimizer.step()
    return loss.item()

def train_on_epoch(model, dataloader, loss_fn, optimizer):
    """
    """
    epoch_loss = 0
    running_loss = 0
    for i, batch in enumerate(dataloader):
        inputs, labels = batch
        new_loss = train_on_minibatch(
            model = model,
            inputs = inputs, 
            labels = labels,
            loss_fn = loss_fn,
            optimizer = optimizer
        )
        running_loss += new_loss
        epoch_loss += new_loss
        if i % 10 == 9:    # print every 10 mini-batches
            print('Mean of loss reduction on minibatch {}: {}'.format(i+1, running_loss / 10.0))
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0
    mean_loss = epoch_loss / float(len(dataloader))
    return mean_loss

def get_accuracy(preds, labels, activation):
    """
    """
    probs = activation(preds)
    classes = torch.argmax(probs, dim=1)
    num_correct = sum(classes == labels).item()
    num_examples = preds.shape[0]
    return num_correct / float(num_examples)

def get_accuracy_on_dataloader(model, dataloader, activation):
    """
    """
    running_accuracy = 0.0
    for batch in dataloader:
        inputs, labels = batch
        outputs = model(inputs)
        running_accuracy += get_accuracy(
            preds = outputs,
            labels = labels, 
            activation = activation)
    mean_accuracy = running_accuracy / float(len(dataloader))
    return mean_accuracy

"""
The code below is code that I am not currently using, but mayyyybbbbbeeeee will use later
"""


class NaiveNetSigmoid(nn.Module):
    """
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
                in_channels = 3,
                out_channels = 6,
                kernel_size = (3,3)
                )
        self.conv2 = nn.Conv2d(
                in_channels = 6,
                out_channels = 12,
                kernel_size = (3,3)
                )
        self.conv3 = nn.Conv2d(
                in_channels = 12,
                out_channels = 24,
                kernel_size = (3,3)
                )
        self.conv4 = nn.Conv2d(
                in_channels = 24,
                out_channels = 48,
                kernel_size = (3,3)
                )
        self.pool = nn.MaxPool2d(2,2)
        self.output_activation = nn.Sigmoid()

        self.fc1 = nn.Linear(43200, 4320)
        self.fc2 = nn.Linear(4320, 432)
        self.fc3 = nn.Linear(432, 40)
        self.fc4 = nn.Linear(40, 1)

    def forward(self, x):
        """
        The shape after running 
        input through all the 
        convolution and pooling
        layers is 
        43200
        I computed this manually 
        and am adding it manually
        this is not good practice
        but this is the best 
        I can do right now
        """
        x = F.relu(self.conv1(x))
        #print('Shape after first convolution: {}'.format(x.shape))
        x = self.pool(x)
        #print('Shape after first pool: {}'.format(x.shape))
        x = F.relu(self.conv2(x))
        #print('Shape after second convolution: {}'.format(x.shape))
        x = self.pool(x)
        #print('Shape after second pool: {}'.format(x.shape))
        x = F.relu(self.conv3(x))
        #print('Shape after third convolution: {}'.format(x.shape))
        x = self.pool(x)
        #print('Shape after third pool: {}'.format(x.shape))
        x = F.relu(self.conv4(x))
        #print('Shape after first convolution: {}'.format(x.shape))
        x = self.pool(x)
        #print('Shape after fourth pool: {}'.format(x.shape))
        x = torch.flatten(x, 1)
        #print('shape after flattening: {}'.format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, .2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, .2)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, .2)
        x = self.fc4(x)

        
        return self.output_activation(x)
