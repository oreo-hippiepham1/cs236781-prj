import torch
import torchvision
import math
from torchvision.models import ResNet50_Weights


def prepare_yolov5():   
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, classes=7, autoshape=False, force_reload=True)
    
     # Freeze
    freeze = [f'model.{x}.' for x in range(10)]  # layers to freeze <-- freezing model.0 to model.9 

    for k, v in model.named_parameters():
        v.requires_grad = True  # unfreeze all layers

        if any(x in k for x in freeze): # freezing those in the list
            print(f'freezing {k}')
            v.requires_grad = False
            
    return model


def prepare_retina():
#     retina = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

#     num_classes = 7

#     # replace classification layer 
#     in_features = retina.head.classification_head.conv[0].in_channels
#     num_anchors = retina.head.classification_head.num_anchors
#     retina.head.classification_head.num_classes = num_classes

#     cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1) # out_features = n_anchors * n_classes
#     torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
#     torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
#     # assign cls head to model
#     retina.head.classification_head.cls_logits = cls_logits

    retina = torchvision.models.detection.retinanet_resnet50_fpn(weights_backbone=ResNet50_Weights.DEFAULT, num_classes=7, trainable_backbone_layers=0) # frozen backbone

    return retina
    




