import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular import data_setup  # , engine
from going_modular.helper_functions import download_data, set_seeds, plot_loss_curves
from trainer import PatchEmbedding
from going_modular import engine


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    set_seeds()

    # 1. Set patch size
    patch_size = 16

    # 2. Print shape of original image tensor and get the image dimensions
    print(f"Image tensor shape: {image.shape}")
    height, width = image.shape[1], image.shape[2]

    # 3. Get image tensor and add batch dimension
    x = image.unsqueeze(0)
    print(f"Input image with batch dimension shape: {x.shape}")

    # 4. Create patch embedding layer
    patch_embedding_layer = PatchEmbedding(
        in_channels=3, patch_size=patch_size, embedding_dim=768
    )

    # 5. Pass image through patch embedding layer
    patch_embedding = patch_embedding_layer(x)
    print(f"Patching embedding shape: {patch_embedding.shape}")

    # 6. Create class token embedding
    batch_size = patch_embedding.shape[0]
    embedding_dimension = patch_embedding.shape[-1]
    class_token = nn.Parameter(
        torch.ones(batch_size, 1, embedding_dimension), requires_grad=True
    )  # make sure it's learnable
    print(f"Class token embedding shape: {class_token.shape}")

    # 7. Prepend class token embedding to patch embedding
    patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
    print(
        f"Patch embedding with class token shape: {patch_embedding_class_token.shape}"
    )

    # 8. Create position embedding
    number_of_patches = int((height * width) / patch_size**2)
    position_embedding = nn.Parameter(
        torch.ones(1, number_of_patches + 1, embedding_dimension), requires_grad=True
    )  # make sure it's learnable

    # 9. Add position embedding to patch embedding with class token
    patch_and_position_embedding = patch_embedding_class_token + position_embedding
    print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")

    # Create an instance of MSABlock
    multihead_self_attention_block = MultiheadSelfAttentionBlock(
        embedding_dim=768, num_heads=12  # from Table 1
    )  # from Table 1

    # Pass patch and position image embedding through MSABlock
    patched_image_through_msa_block = multihead_self_attention_block(
        patch_and_position_embedding
    )
    print(f"Input shape of MSA block: {patch_and_position_embedding.shape}")
    print(f"Output shape MSA block: {patched_image_through_msa_block.shape}")

    # Create an instance of MLPBlock
    mlp_block = MLPBlock(
        embedding_dim=768, mlp_size=3072, dropout=0.1  # from Table 1  # from Table 1
    )  # from Table 3

    # Pass output of MSABlock through MLPBlock
    patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
    print(f"Input shape of MLP block: {patched_image_through_msa_block.shape}")
    print(f"Output shape MLP block: {patched_image_through_mlp_block.shape}")

    # Create an instance of TransformerEncoderBlock

    transformer_encoder_block = TransformerEncoderBlock()
    # Create the same as above with torch.nn.TransformerEncoderLayer()

    torch_transformer_encoder_layer = nn.TransformerEncoderLayer(
        d_model=768,  # Hidden size D from Table 1 for ViT-Base
        nhead=12,  # Heads from Table 1 for ViT-Base
        dim_feedforward=3072,  # MLP size from Table 1 for ViT-Base
        dropout=0.1,  # Amount of dropout for dense layers from Table 3 for ViT-Base
        activation="gelu",  # GELU non-linear activation
        batch_first=True,  # Do our batches come first?
        norm_first=True,
    )  # Normalize first or after MSA/MLP layers?
    # Example of creating the class embedding and expanding over a batch dimension

    batch_size = 32
    class_token_embedding_single = nn.Parameter(
        data=torch.randn(1, 1, 768)
    )  # create a single learnable class token
    class_token_embedding_expanded = class_token_embedding_single.expand(
        batch_size, -1, -1
    )  # expand the single learnable class token across the batch dimension, "-1" means to "infer the dimension"

    # Print out the change in shapes
    print(
        f"Shape of class token embedding single: {class_token_embedding_single.shape}"
    )
    print(
        f"Shape of class token embedding expanded: {class_token_embedding_expanded.shape}"
    )
    set_seeds()

    # Create a random tensor with same shape as a single image
    random_image_tensor = torch.randn(
        1, 3, 224, 224
    )  # (batch_size, color_channels, height, width)

    # Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)
    vit = ViT(num_classes=len(class_names))

    # Pass the random image tensor to our ViT instance
    vit(random_image_tensor)

    # Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper
    optimizer = torch.optim.Adam(
        params=vit.parameters(),
        lr=3e-3,  # Base LR from Table 3 for ViT-* ImageNet-1k
        betas=(
            0.9,
            0.999,
        ),  # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
        weight_decay=0.3,
    )  # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set the seeds
    set_seeds()

    # Train the model and save the training results to a dictionary
    results = engine.train(
        model=vit,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device,
    )
    plot_loss_curves(results)
    # save model
    torch.save(vit.state_dict(), "models/vit.pth")


# # Print an input and output summary of our Transformer Encoder (uncomment for full output)
# summary(model=transformer_encoder_block,
#         input_size=(1, 197, 768), # (batch_size, num_patches, embedding_dimension)
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])


# 1. Create a class that inherits from nn.Module
class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short)."""

    # 2. Initialize the class with hyperparameters from Table 1
    def __init__(
        self,
        embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
        num_heads: int = 12,  # Heads from Table 1 for ViT-Base
        attn_dropout: float = 0,
    ):  # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )  # does our batch dimension come first?

    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x,  # query embeddings
            key=x,  # key embeddings
            value=x,  # value embeddings
            need_weights=False,
        )  # do we need the weights or just the layer outputs?
        return attn_output


# 1. Create a class that inherits from nn.Module
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""

    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(
        self,
        embedding_dim: int = 768,  # Hidden Size D from Table 1 for ViT-Base
        mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
        dropout: float = 0.1,
    ):  # Dropout from Table 3 for ViT-Base
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),  # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=mlp_size,  # needs to take same in_features as out_features of layer above
                out_features=embedding_dim,
            ),  # take back to embedding_dim
            nn.Dropout(
                p=dropout
            ),  # "Dropout, when used, is applied after every dense layer.."
        )

    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


# 1. Create a class that inherits from nn.Module
class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""

    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(
        self,
        embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
        num_heads: int = 12,  # Heads from Table 1 for ViT-Base
        mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
        mlp_dropout: float = 0.1,  # Amount of dropout for dense layers from Table 3 for ViT-Base
        attn_dropout: float = 0,
    ):  # Amount of dropout for attention layers
        super().__init__()

        # 3. Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        )

        # 4. Create MLP block (equation 3)
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )

    # 5. Create a forward() method
    def forward(self, x):
        # 6. Create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x) + x

        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x


# 1. Create a ViT class that inherits from nn.Module
class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""

    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(
        self,
        img_size: int = 224,  # Training resolution from Table 3 in ViT paper
        in_channels: int = 3,  # Number of channels in input image
        patch_size: int = 16,  # Patch size
        num_transformer_layers: int = 12,  # Layers from Table 1 for ViT-Base
        embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
        mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
        num_heads: int = 12,  # Heads from Table 1 for ViT-Base
        attn_dropout: float = 0,  # Dropout for attention projection
        mlp_dropout: float = 0.1,  # Dropout for dense/MLP layers
        embedding_dropout: float = 0.1,  # Dropout for patch and position embeddings
        num_classes: int = 1000,
    ):  # Default for ImageNet but can customize this
        super().__init__()  # don't forget the super().__init__()!

        # 3. Make the image size is divisble by the patch size
        assert (
            img_size % patch_size == 0
        ), f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # 4. Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2

        # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True
        )

        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True
        )

        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 8. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # 10. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    # 11. Create a forward() method
    def forward(self, x):
        # 12. Get batch size
        batch_size = x.shape[0]

        # 13. Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(
            batch_size, -1, -1
        )  # "-1" means to infer the dimension (try this line on its own)

        # 14. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1)
        x = self.position_embedding + x

        # 17. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0])  # run on each sample in a batch at 0 index

        return x


if __name__ == "__main__":
    main()
