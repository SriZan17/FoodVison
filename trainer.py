import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular import data_setup  # , engine
from going_modular.helper_functions import download_data, set_seeds, plot_loss_curves


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device
    # Download pizza, steak, sushi images from GitHub
    image_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi",
    )
    image_path
    # Setup directory paths to train and test images
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Create image size (from Table 3 in the ViT paper)
    IMG_SIZE = 224

    # Create transform pipeline manually
    manual_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )
    # Set the batch size

    BATCH_SIZE = (
        1024  # this is lower than the ViT paper but it's because we're starting small
    )

    # Create data loaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms,  # use manually created transforms
        batch_size=BATCH_SIZE,
    )
    # Get a batch of images
    image_batch, label_batch = next(iter(train_dataloader))

    # Get a single image from the batch
    image, label = image_batch[0], label_batch[0]

    # Create example values
    height = 224  # H ("The training resolution is 224.")
    width = 224  # W
    color_channels = 3  # C
    patch_size = 16  # P

    # Calculate N (number of patches)
    number_of_patches = int((height * width) / patch_size**2)
    print(
        f"Number of patches (N) with image height (H={height}), width (W={width}) and patch size (P={patch_size}): {number_of_patches}"
    )

    # Input shape (this is the size of a single image)
    embedding_layer_input_shape = (height, width, color_channels)

    # Output shape
    embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)

    print(f"Input shape (single 2D image): {embedding_layer_input_shape}")
    print(
        f"Output shape (single 2D image flattened into patches): {embedding_layer_output_shape}"
    )

    # set the patch size
    PATCH_SIZE = 16

    # create the conv2d patch embedding layer
    conv2d = nn.Conv2d(
        in_channels=3,
        out_channels=768,
        kernel_size=PATCH_SIZE,
        stride=PATCH_SIZE,
        padding=0,
    )
    # create flatten layer
    flatten = nn.Flatten(
        start_dim=2,
        end_dim=3,
    )

    # 2. Turn image into feature maps
    image_out_of_conv = conv2d(
        image.unsqueeze(0)
    )  # add batch dimension to avoid shape errors
    print(f"Image feature map shape: {image_out_of_conv.shape}")

    # 3. Flatten the feature maps
    image_out_of_conv_flattened = flatten(image_out_of_conv)
    print(f"Flattened image feature map shape: {image_out_of_conv_flattened.shape}")
    # Get flattened image patch embeddings in right shape
    image_out_of_conv_flattened_reshaped = image_out_of_conv_flattened.permute(
        0, 2, 1
    )  # [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
    print(
        f"Patch embedding sequence shape: {image_out_of_conv_flattened_reshaped.shape} -> [batch_size, num_patches, embedding_size]"
    )
    set_seeds(43)
    # Create an instance of patch embedding layer

    patchify = PatchEmbedding(in_channels=3, patch_size=16, embedding_dim=768)
    # Pass a single image through
    print(f"Input image shape: {image.unsqueeze(0).shape}")
    patch_embedded_image = patchify(
        image.unsqueeze(0)
    )  # add an extra batch dimension on the 0th index, otherwise will error
    print(f"Output patch embedding shape: {patch_embedded_image.shape}")

    # Get the batch size and embedding dimension
    batch_size = patch_embedded_image.shape[0]
    embedding_dimension = patch_embedded_image.shape[-1]

    # Create the class token embedding as a learnable parameter that shares the same size as the embedding dimension (D)
    class_token = nn.Parameter(
        torch.ones(
            batch_size, 1, embedding_dimension
        ),  # [batch_size, number_of_tokens, embedding_dimension]
        requires_grad=True,
    )  # make sure the embedding is learnable

    # Show the first 10 examples of the class_token
    print(class_token[:, :, :10])

    # Print the class_token shape
    print(
        f"Class token shape: {class_token.shape} -> [batch_size, number_of_tokens, embedding_dimension]"
    )

    # Add the class token embedding to the front of the patch embedding
    patch_embedded_image_with_class_embedding = torch.cat(
        (class_token, patch_embedded_image), dim=1
    )  # concat on first dimension

    # Print the sequence of patch embeddings with the prepended class token embedding
    print(patch_embedded_image_with_class_embedding)
    print(
        f"Sequence of patch embeddings with class token prepended shape: {patch_embedded_image_with_class_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]"
    )

    # Calculate N (number of patches)

    number_of_patches = int((height * width) / patch_size**2)

    # Get embedding dimension
    embedding_dimension = patch_embedded_image_with_class_embedding.shape[2]

    # Create the learnable 1D position embedding
    position_embedding = nn.Parameter(
        torch.ones(1, number_of_patches + 1, embedding_dimension), requires_grad=True
    )  # make sure it's learnable

    # Show the first 10 sequences and 10 position embedding values and check the shape of the position embedding
    print(position_embedding[:, :10, :10])
    print(
        f"Position embeddding shape: {position_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]"
    )
    # Add the position embedding to the patch and class token embedding

    patch_and_position_embedding = (
        patch_embedded_image_with_class_embedding + position_embedding
    )
    print(patch_and_position_embedding)
    print(
        f"Patch embeddings, class token prepended and positional embeddings added shape: {patch_and_position_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]"
    )


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    # 2. Initialize the class with appropriate variables
    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768
    ):
        super().__init__()

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(
            start_dim=2,  # only flatten the feature map dimensions into a single vector
            end_dim=3,
        )

    # 5. Define the forward method
    def forward(self, x, patch_size=16):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert (
            image_resolution % patch_size == 0
        ), f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        return x_flattened.permute(
            0, 2, 1
        )  # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


if __name__ == "__main__":
    main()
