# Download custom image
import requests
import torch
from pathlib import Path
from going_modular.predictions import pred_and_plot_image
import torchvision
import torch.nn as nn

# Setup custom image path
custom_image_path = Path("data/04-pizza-dad.jpeg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get(
            "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg"
        )
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

class_names = ["Pizza", "Steak", "Sushi"]
model = torchvision.models.vit_b_16()
model.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(DEVICE)
model = model.to(DEVICE)
# model = torch.compile(model)
state_dict = torch.load(
    "models/08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth"
)
# state_dict["heads.head.weight"] = state_dict.pop("heads.weight")
# state_dict["heads.head.bias"] = state_dict.pop("heads.bias")
model.load_state_dict(state_dict)

# Predict on custom image
pred_and_plot_image(model=model, image_path=custom_image_path, class_names=class_names)
