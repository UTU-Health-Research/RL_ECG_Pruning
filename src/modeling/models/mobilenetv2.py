import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchinfo import summary

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v
    
    

class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=False)

    def forward(self, x):
        return self.depthwise(x)
        

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, kernel_size, dropout_rate, block_id):
        super(InvertedResidualBlock, self).__init__()
                
        #assert stride in [1, 2]
        
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        
        mid_channels = in_channels * expansion # equivalent to hidden_dim 
        padding = kernel_size // 2 if stride == 1 else 0

        layers = []
        if block_id != 0:
            # Expansion
            layers.extend([
                nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU6(inplace=True)
            ])
            

        # Depthwise
        layers.extend([
            DepthwiseConv1D(mid_channels, kernel_size, stride, padding if stride == 1 else 0),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout_rate)
        ])

                
        # Projection
        layers.extend([
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])

        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out

class MobileNetV2_1D(nn.Module): 
    def __init__(self, input_channels=12, alpha=1.0, num_classes=10, ag_dim=3,
                 stride_size=4, kernel_size=9, dropout_rate=0.3, include_top=True, pooling=None):
        
        super(MobileNetV2_1D, self).__init__()
        if isinstance(stride_size, int):
            stride_size = [stride_size] * 5
        elif len(stride_size) != 5:
            raise ValueError("stride_size must be an integer or a tuple of length 5.")
        
        #input_channels = input_shape[1]
        self.include_top = include_top
        self.pooling = pooling

        first_block_filters = make_divisible(32 * alpha, 8)
        
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, first_block_filters, kernel_size=3, stride=stride_size[0], padding=1, bias=False),
            nn.BatchNorm1d(first_block_filters),
            nn.ReLU6(inplace=True)
        )

        block_params = [
            (16, 1, 1), (24, stride_size[1], 6), (24, 1, 6),
            (32, stride_size[2], 6), (32, 1, 6), (32, 1, 6),
            (64, stride_size[3], 6), (64, 1, 6), (64, 1, 6), (64, 1, 6),
            (96, 1, 6), (96, 1, 6), (96, 1, 6),
            (160, stride_size[4], 6), (160, 1, 6), (160, 1, 6),
            (320, 1, 6)
        ]
        

        blocks = []
        in_channels = first_block_filters
        
        for i, (filters, stride, expansion) in enumerate(block_params):
            out_channels = make_divisible(filters * alpha, 8)
            blocks.append(
                InvertedResidualBlock(in_channels, out_channels, stride, expansion, kernel_size, 0.1, i)
            )
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks) # features
        
        
        last_block_filters = make_divisible(1280 * alpha, 8) if alpha > 1.0 else 1280
        self.final = nn.Sequential(
            nn.Conv1d(in_channels, last_block_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(last_block_filters),
            nn.ReLU6(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1) if include_top or pooling == 'avg' else (
            nn.AdaptiveMaxPool1d(1) if pooling == 'max' else nn.Identity()
        )

        self.ag_dim = 3  # e.g., [age, gender, etc.]
        self.fc1 = nn.Linear(self.ag_dim, 10)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(last_block_filters + 10, num_classes)
        
         

    def forward(self, x, ag=None):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        x = self.global_pool(x)
        
        x = torch.flatten(x, 1)
        
        if self.include_top:
            x = self.dropout(x)
            if ag is not None:
                ag = self.fc1(ag)           # (batch_size, 64)
                x = torch.cat((x, ag), dim=1)

            x = self.classifier(x)

        return x
        
        

def mobilenetv2_1d(**kwargs):
    """
    Constructs a 1D MobileNetV2 model with age/gender fusion.
    Accepts additional keyword arguments passed to MobileNetV2_1D_AG.
    """
    model = MobileNetV2_1D(alpha=1.0, stride_size=(2, 2, 2, 2, 2), kernel_size=9, **kwargs)
    # Show summary (PyTorch uses channels-first: (batch, channels, length))
    #summary(model, input_size=(1, 12, 4096))  # batch_size=1
    #summary(model, input_data=(torch.randn(1, 12, 4096), torch.randn(1, 3)))
    return model
    
