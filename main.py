import torch
from torch import nn, optim

from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from pathlib import Path
import matplotlib.pyplot as plt

DATASET_PATH = "" # YOUR PATH

WIDTH = 128
HEIGHT = 64
BATCH_SIZE = 64
PIXELS = WIDTH*HEIGHT # 1600x1200 - 128x64

MODELS_PATH = Path("models")
DISCRIMINATOR_PATH = MODELS_PATH / "image_discriminator.1.0.pth"
GENERATOR_PATH = MODELS_PATH / "image_generator.1.0.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_CHANNELS = 3
FEATURE_MULTIPLIER = 64
HIDDEN_DIM = 100

# load data
def load_data() -> DataLoader:
    data_path = Path(DATASET_PATH)
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.RandomVerticalFlip(1.0), # because training dataset is flipped
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.RandomAffine(degrees=0, shear=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    torch.manual_seed(67)
    data = datasets.ImageFolder(root=data_path, transform=transform)
    return DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)


# models
def weights_init(m): # initialize random weights
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, feature_multiplier=FEATURE_MULTIPLIER, color_channels=COLOR_CHANNELS):
        super().__init__()

        self.hidden1 = nn.Sequential(
            nn.Conv2d(color_channels, feature_multiplier, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_multiplier),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(feature_multiplier, feature_multiplier*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*2),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(feature_multiplier*2, feature_multiplier*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*4),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(feature_multiplier*4, feature_multiplier*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*8),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden5 = nn.Sequential(
            nn.Conv2d(feature_multiplier*8, feature_multiplier*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*16),
            nn.LeakyReLU(0.2, True),
        )
        self.out = nn.Conv2d(feature_multiplier*16, 1, (2, 4), 1, 0, bias=False)

    def forward(self, x):
        return self.out(self.hidden5(self.hidden4(self.hidden3(self.hidden2(self.hidden1(x))))))

class Generator(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, feature_multiplier=FEATURE_MULTIPLIER, color_channels=COLOR_CHANNELS):
        super().__init__()

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, feature_multiplier*16, 4, 1, 0, bias=False),  # increased channels
            nn.BatchNorm2d(feature_multiplier*16),
            nn.ReLU(True),
        )
        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*16, feature_multiplier*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*8),
            nn.ReLU(True),
        )
        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*8, feature_multiplier*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*4),
            nn.ReLU(True),
        )
        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*4, feature_multiplier*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*2),
            nn.ReLU(True),
        )
        self.hidden5 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*2, feature_multiplier, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier),
            nn.ReLU(True),
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier,  color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.out(self.hidden5(self.hidden4(self.hidden3(self.hidden2(self.hidden1(x))))))


# training fns
def train_discriminator(discriminator, optimizer, loss_fn, real_data, fake_data, batch_size):
    # real data
    pred_real = discriminator(real_data).view(-1)
    loss_real = loss_fn(pred_real, torch.ones_like(pred_real))

    # fake data
    pred_fake = discriminator(fake_data).view(-1)
    loss_fake = loss_fn(pred_fake, torch.zeros_like(pred_fake))

    optimizer.zero_grad()
    (loss_real + loss_fake).backward()
    optimizer.step()

    return (loss_real + loss_fake).item()

def train_generator(discriminator, optimizer, loss_fn, fake_data, batch_size):
    pred = discriminator(fake_data).view(-1)
    loss = loss_fn(pred, torch.ones_like(pred))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def train(epochs, dataloader, discriminator, generator, d_optim, g_optim, loss_fn):
    results = { # for plotting
        "d_loss": [],
        "g_loss": [],
    }

    # training
    for epoch in range(epochs):
        for n_batch, data in enumerate(dataloader, 0):
            # discriminator
            real_data = data[0].to(device)
            batch_size = real_data.size(0)

            noise = torch.randn(batch_size, HIDDEN_DIM, 2, 1, device=device)

            fake_data = generator(Variable(noise)).detach().to(device) # detach, so gradients are not calculated for generator

            d_loss = train_discriminator(discriminator, d_optim, loss_fn, real_data, fake_data, batch_size)

            # generator
            fake_data = generator(Variable(noise)).to(device)
            g_loss = train_generator(discriminator, g_optim, loss_fn, fake_data, batch_size)

            # save loss
            results["d_loss"].append(d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss)
            results["g_loss"].append(g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss)

    # plot curves
    plt.plot(range(len(results["d_loss"])), results["d_loss"], label="Discriminator loss")
    plt.plot(range(len(results["g_loss"])), results["g_loss"], label="Generator loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # save model
    save_models(discriminator, generator, DISCRIMINATOR_PATH, GENERATOR_PATH)


# create and save models
def save_models(discriminator: nn.Module, generator: nn.Module, discriminator_path: Path, generator_path: Path):
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(obj=discriminator.state_dict(), f=discriminator_path)
    torch.save(obj=generator.state_dict(), f=generator_path)

def create_model():
    dataloader = load_data()

    # models
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    # random weights
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # optim and loss
    lr = 0.002
    beta1 = 0.5
    beta2 = 0.999
    d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 200
    train(epochs=epochs, dataloader=dataloader, discriminator=discriminator, generator=generator, d_optim=d_optim, g_optim=g_optim, loss_fn=loss_fn)

def get_saved_model() -> nn.Module:
    loaded_generator = Generator()

    if not GENERATOR_PATH.is_file():
        print("First have to train model")
        create_model()

    loaded_generator.load_state_dict(torch.load(f=GENERATOR_PATH, map_location=torch.device(device)))

    return loaded_generator


# TOO CALL
def generate_image():
    generator = get_saved_model()
    noise = torch.randn(1, 1000, 2, 1, device=device)
    image = generator(Variable(noise).detach())

    plt.imshow(image)
    plt.title("Generated image")
    plt.axis("off")
    plt.show()

generate_image()