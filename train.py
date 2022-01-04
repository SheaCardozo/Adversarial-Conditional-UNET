import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from utils import extract_cifar, get_model_sets
from models import CIFAR10Dataset, UNet, final_loss

from random import randint
from argparse import ArgumentParser

def main (args):

    SEED = args.seed
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    CHECKPOINT_INTERVAL = args.checkpoints
    GAMMA = args.gamma


    #torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, test_X, test_y = extract_cifar()
    train_mods, test_mods = get_model_sets(device)

    train = CIFAR10Dataset(X, y)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    #test = CIFAR10Dataset(test_X, test_y)
    #test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    mod = UNet(base = 4, out=3072, start = 1024, cdim=10).to(device)

    optimizer = torch.optim.Adam(mod.parameters(), lr=LR) 

    scheduler = ExponentialLR(optimizer, gamma=GAMMA)

    for epoch in range(NUM_EPOCHS):
        for data in train_loader:

            amod_ind = randint(0, len(train_mods)-1)
            amod = train_mods[amod_ind]

            img, labs = data
            img = img.to(device)
            
            target_p = torch.randint(low=0, high=10, size=labs.shape).to(device)
            target_p = torch.nn.functional.one_hot(target_p, num_classes=10).float()
            
            recon = mod(img.view((-1, 3072)), target_p).view((-1, 3, 32, 32)) + img
            adv_p = amod(recon)
                    
            loss = final_loss(target_p, adv_p, recon, img, weight=1)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
                    

        scheduler.step()
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))

        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save(mod, f"checkpoint_{epoch}.pt")

    torch.save(mod, f"checkpoint_{NUM_EPOCHS}.pt")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--checkpoints", type=int, default=100)

    args = parser.parse_args()
    main(args)