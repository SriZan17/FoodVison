import torch
import torchvision
from torch import nn
from pathlib import Path
from going_modular import data_setup
from going_modular import utils
from going_modular import engine
from going_modular import helper_functions
import json

# from torchinfo import summary


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 10

    # Download pizza, steak, sushi images from GitHub
    data_20_percent_path = helper_functions.download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
        destination="pizza_steak_sushi_20_percent",
    )

    # Setup directory paths to train and test images
    train_dir = data_20_percent_path / "train"
    test_dir = data_20_percent_path / "test"
    vit, vit_transforms = create_vit_model(num_classes=3, seed=43)
    (
        train_dataloader_effnetb2,
        test_dataloader_effnetb2,
        class_names,
    ) = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=vit_transforms,
        batch_size=1024,
    )
    # Setup optimizer

    optimizer = torch.optim.Adam(params=vit.parameters(), lr=1e-3)
    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setup the PyTorch TensorBoard logger
    writer = helper_functions.create_writer(
        experiment_name="20%-data-10-epochs",
        model_name="vit",
        extra=f"{EPOCHS}_epochs",
    )

    # Set seeds for reproducibility and train the model
    helper_functions.set_seeds()
    vit_results = engine.train(
        model=vit.to(DEVICE),
        train_dataloader=train_dataloader_effnetb2,
        test_dataloader=test_dataloader_effnetb2,
        epochs=EPOCHS,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE,
        writer=writer,
    )
    # Save model
    utils.save_model(model=vit, model_name="vit" + ".pth", target_dir="models")
    # Count number of parameters in EffNetB2
    vit_total_params = sum(torch.numel(param) for param in vit.parameters())
    # Get the model size in bytes then convert to megabytes
    pretrained_vit_model_size = Path("models/vit.pth").stat().st_size // (
        1024 * 1024
    )  # division converts bytes to megabytes (roughly)
    print(
        f"Pretrained EffNetB2 feature extractor model size: {pretrained_vit_model_size} MB"
    )
    # Create a dictionary with EffNetB2 statistics
    vit_stats = {
        "test_loss": vit_results["test_loss"][-1],
        "test_acc": vit_results["test_acc"][-1],
        "number_of_parameters": vit_total_params,
        "model_size (MB)": pretrained_vit_model_size,
    }
    # Save the dictionary as a JSON file
    with open("vit_stats.json", "w") as json_file:
        json.dump(vit_stats, json_file)

    helper_functions.plot_loss_curves(vit_results)


def create_vit_model(num_classes: int = 3, seed: int = 43):
    """Creates a ViT-B/16 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of target classes. Defaults to 3.
        seed (int, optional): random seed value for output layer. Defaults to 42.

    Returns:
        model (torch.nn.Module): ViT-B/16 feature extractor model.
        transforms (torchvision.transforms): ViT-B/16 image transforms.
    """
    # Create ViT_B_16 pretrained weights, transforms and model
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

    # Freeze all layers in model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head to suit our needs (this will be trainable)
    torch.manual_seed(seed)
    model.heads = nn.Sequential(
        nn.Linear(
            in_features=768,  # keep this the same as original model
            out_features=num_classes,
        )
    )  # update to reflect target number of classes

    return model, transforms


if __name__ == "__main__":
    main()
