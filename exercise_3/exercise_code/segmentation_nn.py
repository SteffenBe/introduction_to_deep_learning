"""SegmentationNN"""
import torch
import torch.nn as nn

import torchvision.models as models
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, add_layers=True, k_size=5):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        self.num_classes = num_classes

        self.first = models.vgg11(pretrained=True).features

        if add_layers:

            self.second = nn.Sequential(
                nn.Conv2d(512, 1000, k_size),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Conv2d(1000, 1500, 1),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Conv2d(1500, num_classes, 1)
                )

        else:

            self.second = nn.Conv2d(512, num_classes, 1)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        relevant_dims = x.size()[2:]

        x = self.first(x)
        x = self.second(x)
        x = F.upsample(x, relevant_dims, mode='bilinear')

        return x

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
