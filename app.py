import torch
from timeit import default_timer as timer
from typing import Dict, Tuple
from lord import create_effnetb2_model
import gradio as gr
import random
from pathlib import Path


def main():
    # Get a list of all test image filepaths
    test_dir = "data/pizza_steak_sushi/test"
    test_data_paths = list(Path(test_dir).glob("*/*.jpg"))

    # Create a list of example inputs to our Gradio demo
    example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]

    title = "Pizza, Steak, Sushi Classifier"
    description = "An EfficientNetB2 feature extractor computer vision model to"
    description += " classify images of pizza, steak and sushi."
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=3, label="Predictions"),  # what are the outputs?
            gr.Number(label="Prediction time (s)"),
        ],  # our fn has two outputs, therefore we have two outputs
        examples=example_list,
        title=title,
        description=description,
        # article=article,
    )
    demo.launch(debug=False, share=True)


def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    # Start the timer
    start_time = timer()

    device = "cpu"
    class_names = ["Pizza", "Steak", "Sushi"]
    model, model_transforms = create_effnetb2_model(num_classes=len(class_names))
    # model = torch.compile(model)
    model.load_state_dict(torch.load("models/effnetb2.pth"))
    model.to(device)

    # Transform the target image and add a batch dimension
    img = model_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(model(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


if __name__ == "__main__":
    main()
