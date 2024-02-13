import torch
import numpy as np
import argparse
import torch.optim as optim
import torchvision.transforms as T
from UNet import UNET
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
DEVICE = torch.device("mps")
NUM_EPOCHS = 25


def train_loop(loader, model, optimizer, loss, epoch):
    loop = tqdm(loader)
    for batch_idx, (image, mask) in enumerate(loop):
        # print(f'Processing batch {batch_idx}')
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        # forward
        predictions = model(image)
        loss_val = loss(predictions, mask)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        loop.set_postfix(loss=loss_val.item(), epoch=epoch)


def train_and_save_model(model_output_path, checkpoint_path=None):
    model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    epoch = 0
    transform = T.Compose(
        [
            T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ]
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = CarvanaDataset(
        image_dir="./extracted_data/image",
        mask_dir="./extracted_data/mask",
        transform=transform,
    )
    loader = DataLoader(dataset, shuffle=True, batch_size=32)

    if checkpoint_path:
        print("Resuming from checkpoint")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

    for epoch in range(epoch + 1, NUM_EPOCHS):
        train_loop(loader, model, optimizer, loss_fn, epoch)
        if epoch % 2 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                model_output_path,
            )


def load_and_eval(model_path, image_path, output_path):
    model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    img = Image.open(image_path)
    transform = T.Compose(
        [
            T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ]
    )
    img = transform(img).unsqueeze(0).to(DEVICE)
    model_output = model(img)
    pil_img = Image.fromarray(
        (model_output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
    )
    pil_img.save(output_path)


def main(args):
    if args.load_model:
        load_and_eval(args.model_path, args.image_path, args.output_path)
    else:
        if len(args.resume_from_checkpoint) > 0:
            train_and_save_model(
                args.model_path, checkpoint_path=args.resume_from_checkpoint
            )
        else:
            train_and_save_model(args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--load_model", type=bool, default=False, help="Load model")
    parser.add_argument(
        "--model_path", type=str, help="Model weight file path to save or load from"
    )
    parser.add_argument(
        "--image_path", type=str, help="Evaluate model on a single image"
    )
    parser.add_argument("--output_path", type=str, help="Save output to")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="checkpoint path to resume from",
    )
    args = parser.parse_args()
    main(args)
