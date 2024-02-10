import torch
import torch.optim as optim
import torchvision.transforms as T
from UNet import UNET
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
DEVICE = torch.device("cpu")
NUM_EPOCHS = 25

def train_loop(loader, model, optimizer, loss):
    loop = tqdm(loader)
    for batch_idx, (image,mask) in enumerate(loop):
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        #forward
        predictions = model(image)
        loss_val = loss(predictions,mask)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loop.set_postfix(loss = loss_val.item())


def main():
    transform = T.Compose([
        T.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]),
    ])

    model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    dataset = CarvanaDataset(image_dir="./extracted_data/image",mask_dir="./extracted_data/mask", transform=transform)
    loader = DataLoader(dataset,shuffle=True, batch_size=32)

    for epochs in range(NUM_EPOCHS):
        train_loop(loader,model,optimizer,loss_fn)


if __name__ == "__main__":
    main()
