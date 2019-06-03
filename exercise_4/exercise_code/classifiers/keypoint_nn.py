import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self, dropout=0.3):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        pool = 2
        stride_pool = 2

        # self.conv = nn.Sequential(
        #         nn.Conv2d(1, 32, 4),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d(pool, stride_pool),
        #         nn.Dropout(p=dropout),
        #         nn.Conv2d(32, 64, 3),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(p=dropout),
        #         nn.MaxPool2d(pool, stride_pool)
        #         )

        # self.fc = nn.Sequential(
        #         nn.Linear(30976, 8192),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(p=dropout),
        #         nn.Linear(8192, 1024),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(p=dropout),
        #         nn.Linear(1024, 30)
        #         )

        self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 7),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool, stride_pool),
                # nn.Dropout(p=dropout),
                nn.Conv2d(32, 64, 5),
                nn.ReLU(inplace=True),
                # nn.Dropout(p=dropout),
                nn.MaxPool2d(pool, stride_pool)
                )

        self.fc = nn.Sequential(
                nn.Linear(25600, 2560),
                nn.ReLU(inplace=True),
                nn.Linear(2560, 30)
                )

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################

        x = self.conv(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
