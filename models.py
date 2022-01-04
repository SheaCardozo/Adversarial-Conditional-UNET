import torch
from torch.utils.data import Dataset
import torch.nn.functional as F 


class CIFAR10Dataset(Dataset):
    def __init__(self, img_list, lab_list, train=True, transform=None):
        self.images = img_list
        self.labels = lab_list
        self.transform = transform
        self.train = train


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        lab = self.labels[idx]
        imgarr = self.images[idx]
        
        r = torch.Tensor(imgarr[:1024].reshape((32, 32))).unsqueeze(0)
        g = torch.Tensor(imgarr[1024:(1024*2)].reshape((32, 32))).unsqueeze(0)
        b = torch.Tensor(imgarr[(1024*2):(1024*3)].reshape((32, 32))).unsqueeze(0)

        img = torch.cat((r, g, b), axis=0) / 255
        
        if self.transform:
            lab = self.transform(img)
        return img, lab
class LinSkip(torch.nn.Module):
    def __init__(self, dim): 
        super(LinSkip, self).__init__()
        
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(dim),
                torch.nn.Linear(dim, dim),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(dim),
                torch.nn.Linear(dim, dim),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(dim),
        )
    
    def forward(self, x):
        return self.layers(x) + x

class ULayer(torch.nn.Module):
    def __init__(self, udim, ddim, base, cdim=0): 
        super(ULayer, self).__init__()
        
        self.base = (base == 0)
        self.conditional = (cdim != 0)
        
        self.down = torch.nn.Sequential(
                torch.nn.Linear(udim, ddim),
                torch.nn.LeakyReLU(),
                LinSkip(ddim),
                LinSkip(ddim),
            )
        
        
        if self.base:
            self.inner = torch.nn.Identity()
            
            self.up = torch.nn.Sequential(
                    torch.nn.Linear(cdim + ddim, udim),
                    torch.nn.LeakyReLU(),
                    LinSkip(udim),
                    LinSkip(udim),
            )
        else:
            self.inner = ULayer(ddim, ddim // 2, base-1, cdim)
            
            self.up = torch.nn.Sequential(
                    torch.nn.Linear(udim, udim),
                    torch.nn.LeakyReLU(),
                    LinSkip(udim),
                    LinSkip(udim)
            )


        if torch.cuda.is_available():
            self = self.cuda()
    
    def forward(self, x, c=None):
        if not self.conditional:
            assert c is None
            
        xc = self.down(x)
        
        if not self.base:
            x = self.inner(xc, c)
        else:
            x = self.inner(xc)

        if self.base and self.conditional:
            x = torch.cat((x, c), dim=1)
        elif not self.base:
            x = torch.cat((x, xc), dim=1)
        x = self.up(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self, base, out, start, cdim=0): 
        super(UNet, self).__init__()
        
        self.out = out
        
        self.conditional = (cdim != 0)
        
        self.start = torch.nn.Sequential(
                    torch.nn.Linear(out, start),
                    torch.nn.LeakyReLU(),
                    LinSkip(start),
                    LinSkip(start),
            )
        
        self.net = ULayer(start, start // 2, base, cdim)
        
        self.out = torch.nn.Sequential(
                    torch.nn.Linear(start, out),
                    LinSkip(out),
                    LinSkip(out),
                    torch.nn.Linear(out, out),
                    torch.nn.Sigmoid(),
            )

        if torch.cuda.is_available():
            self = self.cuda()
    
    def forward(self, x, c=None):
        if not self.conditional:
            assert c is None
                        
        x = self.start(x)
        x = self.net(x, c)
        x = self.out(x)
        return 2*x - 1


def final_loss(target_p, adv_p, recon, img, weight=1):
    BCE = F.bceloss(F.softmax(adv_p, dim=1), target_p, reduction="mean")
    MSE = F.mseloss(recon, img, reduction="mean")
    return BCE + weight*MSE