from going_modular import helper_functions
import torch
from PIL import Image
from timeit import default_timer as timer
from tqdm.auto import tqdm
from typing import List, Dict
import pathlib
import torchvision
from lord import create_effnetb2_model
import pandas as pd
from duke import create_vit_model


def main():
    DEVICE = "cpu"
    # Download pizza, steak, sushi images from GitHub
    data_20_percent_path = helper_functions.download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
        destination="pizza_steak_sushi_20_percent",
    )
    test_dir = data_20_percent_path / "test"

    # Get all test data paths
    print(f"[INFO] Finding all filepaths ending with '.jpg' in directory: {test_dir}")
    test_data_paths = list(pathlib.Path(test_dir).glob("*/*.jpg"))
    test_data_paths[:5]

    class_names = ["Pizza", "Steak", "Sushi"]
    effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=len(class_names))
    # model = torch.compile(model)
    effnetb2.load_state_dict(torch.load("models/effnetb2.pth"))

    # Make predictions across test dataset with EffNetB2
    effnetb2_test_pred_dicts = pred_and_store(
        paths=test_data_paths,
        model=effnetb2,
        transform=effnetb2_transforms,
        class_names=class_names,
        device=DEVICE,
    )  # make predictions on CPU
    effnetb2_test_pred_df = pd.DataFrame(effnetb2_test_pred_dicts)

    # Find the average time per prediction
    effnetb2_average_time_per_pred = round(
        effnetb2_test_pred_df.time_for_pred.mean(), 4
    )

    vit, vit_transforms = create_vit_model(num_classes=len(class_names))
    # model = torch.compile(model)
    vit.load_state_dict(torch.load("models/vit.pth"))

    # Make predictions across test dataset with EffNetB2
    vit_test_pred_dicts = pred_and_store(
        paths=test_data_paths,
        model=vit,
        transform=vit_transforms,
        class_names=class_names,
        device=DEVICE,
    )  # make predictions on CPU
    vit_test_pred_df = pd.DataFrame(vit_test_pred_dicts)

    # Find the average time per prediction
    vit_average_time_per_pred = round(vit_test_pred_df.time_for_pred.mean(), 4)

    print(
        f"EffNetB2 average time per prediction: {effnetb2_average_time_per_pred} seconds"
    )
    print(f"Vit average time per prediction: {vit_average_time_per_pred} seconds")


# Function to return a list of dictionaries with sample, truth label, prediction, prediction probability
# and prediction time
def pred_and_store(
    paths: List[pathlib.Path],
    model: torch.nn.Module,
    transform: torchvision.transforms,
    class_names: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict]:
    # 2. Create an empty list to store prediction dictionaires
    pred_list = []

    # 3. Loop through target paths
    for path in tqdm(paths):
        # 4. Create empty dictionary to store prediction information for each sample
        pred_dict = {}

        # 5. Get the sample path and ground truth class name
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name

        # 6. Start the prediction timer
        start_time = timer()

        # 7. Open image path
        img = Image.open(path)

        # 8. Transform the image, add batch dimension and put image on target device
        transformed_image = transform(img).unsqueeze(0).to(device)

        # 9. Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()

        # 10. Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            pred_logit = model(transformed_image)  # perform inference on target sample
            pred_prob = torch.softmax(
                pred_logit, dim=1
            )  # turn logits into prediction probabilities
            pred_label = torch.argmax(
                pred_prob, dim=1
            )  # turn prediction probabilities into prediction label
            pred_class = class_names[
                pred_label.cpu()
            ]  # hardcode prediction class to be on CPU

            # 11. Make sure things in the dictionary are on CPU (required for inspecting predictions later on)
            pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict["pred_class"] = pred_class

            # 12. End the timer and calculate time per pred
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time - start_time, 4)

        # 13. Does the pred match the true label?
        pred_dict["correct"] = class_name == pred_class

        # 14. Add the dictionary to the list of preds
        pred_list.append(pred_dict)

    # 15. Return list of prediction dictionaries
    return pred_list


if __name__ == "__main__":
    main()
