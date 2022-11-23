import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from models.base_model import BaseModel
from torchvision.models import maxvit_t, MaxVit_T_Weights

class MaxVit_Model(BaseModel):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model_name='maxvit_t'
        self.model = maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
        num_ftrs = list(self.model.classifier.children())[-1].in_features
        classifier = list(self.model.classifier.children())[:-1]
        self.model.classifier = nn.Sequential(*classifier,nn.Linear(in_features=num_ftrs, out_features=num_classes))

    def forward(self, x):
        x=self.model(x)
        return x

    def load_state_and_eval(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()
