import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from .piece_classes import classes


def load_model_for_prediction():
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load('detection/models/mobilenetv2_chess2.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    # convert image from NumPy to PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # Preprocess the image
    # image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]

    return predicted_class


def classify_all_pieces(pieces, image):
    # the index of pieces dictionary is (piece_name, i) returning the bbox (x1, y1, x2, y2)
    # i is the id of the piece
    # piece_name is the name of the piece

    # load the model
    model = load_model_for_prediction()
    # classify all pieces
    classified_pieces = {}
    for (piece, i), (x1, y1, x2, y2) in pieces.items():
        # get the piece image
        X1, Y1, X2, Y2 = map(int, (x1, y1, x2, y2))
        piece_image = image[Y1:Y2, X1:X2]
        # predict the class
        predicted_class = predict(model, piece_image)
        classified_pieces[(predicted_class, i)] = (x1, y1, x2, y2)
    return classified_pieces
