import torch
import torch.nn as nn


class FermiMultiBranchCNN(nn.Module):
  def __init__(self):
    super().__init__()

    # Define a helper function to create the 3 independent branches
    def create_branch():
      return nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2), # 113 -> 56

          nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2), # 56 -> 28

          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2), # 28 -> 7

          nn.Flatten() # out_channels * 7 * 7 features per branch
      )

    # Create the 3 independent branches
    self.branch_x = create_branch()
    self.branch_y = create_branch()
    self.branch_top = create_branch()

    # Classifier
    self.classifier = nn.Sequential(
        nn.Linear(in_features=3*32*7*7, out_features=128),
        nn.ReLU(),
        nn.Dropout(0.5), # reduces overfitting by randomly omitting half of the feature detectors on each training case
        nn.Linear(in_features=128, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=2) # 2 Outputs: [Proton Score, Electron Score]
    )

  def forward(self, x):
    # The input 'x' is the stacked tensor of shape [Batch, 3, 113, 113]
    # Slice it into 3 separate tensors of shape [Batch, 1, 113, 113]
    view_x = x[:, 0:1, :, :]
    view_y = x[:, 1:2, :, :]
    view_top = x[:, 2:3, :, :]

    # Pass each view through its own separate branch
    out_x = self.branch_x(view_x)
    out_y = self.branch_y(view_y)
    out_top = self.branch_top(view_top)

    # Concatenate the extracted features from the 3 views
    merged_features = torch.cat((out_x, out_y, out_top), dim=1)

    # Pass the flattened vector through the classifier
    output = self.classifier(merged_features)
    return output