import torch
from torch import nn, optim

from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from pathlib import Path
import matplotlib.pyplot as plt

# make it with convolutional layers for RGB generation

WIDTH = 128
HEIGHT = 64
BATCH_SIZE = 32
PIXELS = WIDTH*HEIGHT # 1600x1200 - 128x64

MODELS_PATH = Path("models")
DISCRIMINATOR_PATH = MODELS_PATH / "image_discriminator.1.0.pth"
GENERATOR_PATH = MODELS_PATH / "image_generator.1.0.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_CHANNELS = 3
FEATURE_MULTIPLIER = 64
HIDDEN_DIM = 100

SAMPLES = 16
NOISE = Variable(torch.randn(BATCH_SIZE, HIDDEN_DIM, 2, 1))(SAMPLES)

# models
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
            nn.BatchNorm2d(feature_multiplier*8),
            nn.LeakyReLU(0.2, True),
        )
        self.out = nn.Conv2d(feature_multiplier*16, 1, (4, 2), 1, 0, bias=False),

    def forward(self, x):
        return self.out(self.hidden4(self.hidden3(self.hidden2(self.hidden1(x)))))
    
class Generator(nn.Module):
    def __init__(self, hiddem_dim=HIDDEN_DIM, feature_multiplier=FEATURE_MULTIPLIER, color_channels=COLOR_CHANNELS):
        super().__init__()

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(hiddem_dim, feature_multiplier*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_multiplier*8),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*8, feature_multiplier*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*4),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*4, feature_multiplier*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*2),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*2, feature_multiplier, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier),
            nn.LeakyReLU(0.2, True),
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier,  color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.out(self.hidden4(self.hidden3(self.hidden2(self.hidden1(x)))))
    

# training fns
def train_discriminator(discriminator, optimizer, loss_fn, real_data, fake_data, n):
    # real data
    pred_real = discriminator(real_data)
    loss_real = loss_fn(pred_real, Variable(torch.ones(n, 1)))

    # fake data
    pred_fake = discriminator(fake_data)
    loss_fake = loss_fn(pred_fake, Variable(torch.zeros(n, 1)))

    optimizer.zero_grad()
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()

    return loss_real + loss_fake, pred_real, pred_fake

def train_generator(discriminator, optimizer, loss_fn, fake_data, n):
    pred = discriminator(fake_data)
    loss = loss_fn(pred, Variable(torch.ones(n, 1)))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def train(epochs, dataloader, discriminator, generator, d_optim, g_optim, loss_fn, noise):
    results = { # for plotting
        "d_loss": [],
        "g_loss": [],
    }

    # trianing
    for epoch in range(epochs):
        for n_batch, (real_batch,_) in enumerate(dataloader):
            n = real_batch.size(0)
            real_batch = real_batch.to(device)

            # discriminator
            real_data = Variable(real_batch.view(real_batch.size(0), PIXELS)) # images to vectors
            fake_data = generator(noise(n)).detach() # detach, so gradients are not calculated for generator
            
            d_loss, d_pred_real, d_pred_fake = train_discriminator(discriminator, d_optim, loss_fn, real_data, fake_data, n)

            # generator
            fake_data = generator(noise(n))
            g_loss = train_generator(discriminator, g_optim, loss_fn, fake_data, n)

            # save loss
            results["d_loss"].append(d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss)
            results["g_loss"].append(g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss)

    # plot curves
    plt.plot(epochs, d_loss, label="d_loss")
    plt.plot(epochs, g_loss, label="g_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # save model
    save_models(discriminator, generator, DISCRIMINATOR_PATH, GENERATOR_PATH)


# save models
def save_models(discriminator: nn.Module, generator: nn.Module, discriminator_path: Path, generator_path: Path):
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(obj=discriminator.state_dict(), f=discriminator_path)
    torch.save(obj=generator.state_dict(), f=generator_path)

# load data
def load_data() -> DataLoader:
    data_path = Path("D:\Code\Code\Languages\Python\AI_ML\Actual Projects\Image generator\ImageGenerationFromScratch\data\Hands")
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.RandomVerticalFlip(1.0), # because training dataset is flipped
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.RandomAffine(degrees=0, shear=15),
        transforms.ToTensor()
    ])

    torch.manual_seed(67)
    data = datasets.ImageFolder(root=data_path, transform=transform)
    return DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

def create_model():
    dataloader = load_data()

    # models
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    # optim and loss
    d_optim = optim.Adam()
    g_optim = optim.Adam()

    loss_fn = nn.BCELoss()
    
    epochs = 200
    train(epochs=epochs, dataloader=dataloader, discriminator=discriminator, generator=generator, d_optim=d_optim, g_optim=g_optim, loss_fn=loss_fn, noise=NOISE)


# TOO CALL
def get_saved_model() -> nn.Module:
    loaded_generator = Generator()

    if not GENERATOR_PATH.is_file():
        print("First have to train model")
        create_model()

    loaded_generator.load_state_dict(torch.load(f=GENERATOR_PATH, map_location=torch.device(device)))

    return loaded_generator

def generate_image():
    generator = get_saved_model()
    image = generator(Variable(NOISE(1)).detach())

    plt.imshow(image)
    plt.title("Generated image")
    plt.axis("off")
    plt.show()

# generate_image()