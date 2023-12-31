# The following requires torch v0.12+ and torchvision v0.13+
import torch
import torchvision
import torch.nn as nn
from going_modular.helper_functions import set_seeds
from going_modular.helper_functions import download_data
from going_modular import data_setup
from going_modular import engine
from going_modular.helper_functions import plot_loss_curves
from going_modular.helper_functions import create_writer
from going_modular.utils import save_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 10
    batch_size = 1024

    # 1. Get pretrained weights for ViT-Base
    pretrained_vit_weights = (
        torchvision.models.ViT_B_16_Weights.DEFAULT
    )  # requires torchvision >= 0.13, "DEFAULT" means best available

    # 2. Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(
        device
    )

    # 3. Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Download pizza, steak, sushi images from GitHub
    image_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi",
    )

    # Setup train and test directory paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Get automatic transforms from pretrained ViT weights
    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    # Setup dataloaders
    (
        train_dataloader_pretrained,
        test_dataloader_pretrained,
        class_names,
    ) = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=pretrained_vit_transforms,
        batch_size=batch_size,
    )

    # 4. Change the classifier head (set the seeds to ensure same initialization with linear head)
    set_seeds()
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(
        device
    )
    # pretrained_vit # uncomment for model output

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the classifier head of the pretrained ViT feature extractor model
    set_seeds()
    writer = create_writer(
        model_name="pretrained_vit",
        experiment_name="feature_extractor",
        extra=str(num_epochs) + "_epochs",
    )
    pretrained_vit_results = engine.train(
        model=pretrained_vit,
        train_dataloader=train_dataloader_pretrained,
        test_dataloader=test_dataloader_pretrained,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=num_epochs,
        device=device,
        writer=writer,
    )
    plot_loss_curves(pretrained_vit_results)
    save_model(
        model=pretrained_vit,
        target_dir="models",
        model_name="08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth",
    )


if __name__ == "__main__":
    main()
