# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:42:53 2020

@author: flash
"""
from facenet_pytorch import InceptionResnetV1
import torch
import torchvision

resnet = InceptionResnetV1(pretrained='vggface2').eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(resnet, example)
traced_script_module.save("resnet.pt")