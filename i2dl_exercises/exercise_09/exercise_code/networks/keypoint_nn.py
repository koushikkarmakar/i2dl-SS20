"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super().__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################


        self.segmentation = nn.Sequential(
            nn.Conv2d(in_channels = 1,
                      out_channels = 32,
                      kernel_size = 3,
                      stride = 1,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2
                         ),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels= 32, 
                      out_channels = 64,
                      kernel_size = 3,
                      stride = 1,
        
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2
                         ),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels= 64, 
                      out_channels = 128,
                      kernel_size = 2,
                      stride = 1,
        
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2
                         ),
            nn.Dropout(p=0.3),
            nn.Conv2d(in_channels= 128, 
                      out_channels = 256,
                      kernel_size = 1,
                      stride = 1,
        
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2
                         ),
            nn.Dropout(p=0.4),     
        )

        self.linear = nn.Sequential(
            nn.Linear(6400, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 30)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        if (x.shape == torch.Size([1, 96, 96])):
            x = x.unsqueeze(0)
        
        x = self.segmentation(x)
        cnn_output_size = x.size()[1] * x.size()[2] * x.size()[3]
        
        x = x.view(-1, cnn_output_size)

        x = self.linear(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x
    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################
        optim = optimizer.Adam(self.parameters(), lr=self.hparams["learning_rate"], betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return optim


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
