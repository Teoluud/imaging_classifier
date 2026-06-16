import torch
import torch.nn as nn


class FermiMultiBranchCNN(nn.Module):
    """ Multi-branch network classifier to analyze 3 projections independentely.
    """

    def __init__(self) -> None:
        super().__init__()
        # Construct three identical independent layout structures
        self.branch_x = self._create_branch()
        self.branch_y = self._create_branch()
        self.branch_top = self._create_branch()

        # Connected classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features=3 * 32 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2)  # 2 Outputs: [Proton Score, Electron Score]
        )

    def _create_branch(self) -> nn.Sequential:
        """ Constructs an individual view's processing convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 113 -> 56

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 56 -> 28

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28 -> 7

            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Separate individual spatial orientation profiles from channels
        view_x = x[:, 0:1, :, :]
        view_y = x[:, 1:2, :, :]
        view_top = x[:, 2:3, :, :]

        # Forward passes across distinct network streams
        out_x = self.branch_x(view_x)
        out_y = self.branch_y(view_y)
        out_top = self.branch_top(view_top)

        # Merge views and project through fully connected classifier layers
        merged_features = torch.cat((out_x, out_y, out_top), dim=1)
        output = self.classifier(merged_features)
        return output
    

class FermiSingleBranchCNN(nn.Module):
    """ Single-branch classifier (conventional CNN).
    """

    def __init__(self) -> None:
        super().__init__()

        self.convolutional_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3*8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 113 -> 56

            nn.Conv2d(in_channels=3*8, out_channels=3*16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 56 -> 28

            nn.Conv2d(in_channels=3*16, out_channels=3*32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28 -> 7

            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=3 * 32 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2)  # 2 Outputs: [Proton Score, Electron Score]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.convolutional_stack(x))

class FermiMeritVarsNN(nn.Module):
    """ Simple classifier for Merit Variables.
    """

    def __init__(self) -> None:
        super().__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=17, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=2)

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_stack(x)
        