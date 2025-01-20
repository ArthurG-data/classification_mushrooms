import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        y = self.fc2(x)
        return y

class MyClassifier:
    def __init__(self):
        self.class_labels = ['edible_1', 'edible_2', 'edible_3', 'edible_4', 'edible_5',
                             'poisonous_1', 'poisonous_2', 'poisonous_3', 'poisonous_4', 'poisonous_5']


    def model(self):
        model = Model()
        model.load_state_dict(torch.load('deep__dyno_ADAM_70_30_b12_0_0001_50_AUGM_877_best.pth', weights_only=True))
        return model

    def setup(self):
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()
        self.model = self.model()
        self.model.eval()
        
        imagenet_means = (0.485, 0.456, 0.406)
        imagenet_stds = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_means, imagenet_stds),
        ])

    def test_image(self, image):
        ims = self.transform(image).unsqueeze(0)
        features = self.dino(ims)
        predicted_cls = self.model(features)
        predicted_index = torch.argmax(predicted_cls, axis = 1)
        return self.class_labels[predicted_index]        
        
