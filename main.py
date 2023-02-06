from torchvision import transforms
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.manual_seed(0)
DIM_A = 3
DIM_B = 3
LOAD_SHAPE = 286
TARGET_SHAPE = 256

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")




def display_images(image):
    image = (image + 1) / 2
    shift_image = image
    new_image = shift_image.detach()
    new_image = new_image.cpu()
    new_image = new_image.view(-1, *(3, 256, 256))
    grid = make_grid(new_image[:25], nrow=5)

    final_image = grid.permute(1, 2, 0).squeeze()
    final_image = final_image.cpu()
    final_image = final_image.detach()

    return final_image.numpy()



class ImageDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform


    def __getitem__(self, index):

        # image = self.transform(Image.open(self.image))
        image = self.transform(self.image)
        if image.shape[0] != 3: 
            image = image.repeat(3, 1, 1)

        return (image - 0.5) * 2

    def __len__(self):
        return 1 



class ResidualBlock(nn.Module):

    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()
 
    def forward(self, x):

        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x




class ContractingBlock(nn.Module):

    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn
 
    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x
 
class ExpandingBlock(nn.Module):

    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
 
    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x
 
class FeatureMapBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')
 
    def forward(self, x):

        x = self.conv(x)
        return x



class Generator(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = torch.nn.Tanh()
 
    def forward(self, x):

        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)


GEN_AB = Generator(DIM_A, DIM_B).to(DEVICE)

pre_dict = torch.load('horse2zebra.pth', map_location=torch.device(DEVICE))
GEN_AB.load_state_dict(pre_dict['gen_AB'])



transform = transforms.Compose([
    transforms.Resize(LOAD_SHAPE),
    transforms.RandomCrop(TARGET_SHAPE),
    transforms.ToTensor(),
])

def predict(upload_image):
    dataset = ImageDataset(upload_image, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for real_image in tqdm(dataloader):
        pass
    with torch.no_grad():
        real_image = real_image.to(DEVICE)
        output_image = GEN_AB(real_image)

        return real_image, output_image 