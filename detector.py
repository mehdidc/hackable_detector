import torch.nn as nn
import torch


class DetectorBuilder:

    def __init__(self):
        self.layers = []
        self.scales = []
        self.branch_layer_index_for_scale = []
        self.aspect_ratios_for_scale = []

    def add_layers_from(self, model):
        self.layers.append(model)
    
    def add_layers(self, layers):
        self.layers.extend(layers)

    def add_prediction_branch(self, scale, aspect_ratios):
        self.scales.append(scale)
        self.branch_layer_index_for_scale.append(len(self.layers) - 1)
        self.aspect_ratios_for_scale.append(aspect_ratios)
    
    def build(self, **options):
        return Detector(
            layers=self.layers, 
            scales=self.scales,
            branch_layer_index_for_scale=self.branch_layer_index_for_scale,
            aspect_ratios_for_scale=self.aspect_ratios_for_scale,
            **options,
        )


def default_prediction_layer(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3, 
        padding=1,
    ) 


class Detector(nn.Module):

    def __init__(self, layers, scales, branch_layer_index_for_scale, aspect_ratios_for_scale, nb_classes, nb_coords=4, prediction_layer_func=default_prediction_layer):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.scales = scales
        self.branch_layer_index_for_scale = branch_layer_index_for_scale
        self.aspect_ratios_for_scale = aspect_ratios_for_scale
        self.nb_classes = nb_classes
        self.nb_coords = nb_coords
        self.prediction_layer_func = prediction_layer_func

        self.bounding_box_prediction_layers = nn.ModuleList([
            prediction_layer_func(layers[layer_index].out_channels, nb_coords * len(aspect_ratios))
            for layer_index, aspect_ratios in zip(branch_layer_index_for_scale, aspect_ratios_for_scale)
        ])
        self.class_prediction_layers = nn.ModuleList([
            prediction_layer_func(layers[layer_index].out_channels, nb_classes * len(aspect_ratios))
            for layer_index, aspect_ratios in zip(branch_layer_index_for_scale, aspect_ratios_for_scale)
        ])

    def predict_raw(self, X):
        cur_branch_index = 0
        next_layer_index = self.branch_layer_index_for_scale[cur_branch_index]
        h = X
        boxes = []
        classes = []
        for layer_index, layer in enumerate(self.layers):
            h = layer(h)
            if layer_index == next_layer_index:
                bounding_box_prediction_layer = self.bounding_box_prediction_layers[cur_branch_index]
                class_prediction_layer = self.class_prediction_layers[cur_branch_index]
                box_output, class_output = bounding_box_prediction_layer(h), class_prediction_layer(h)
                boxes.append(box_output)
                classes.append(class_output)
                cur_branch_index += 1
                if cur_branch_index < len(self.branch_layer_index_for_scale):
                    next_layer_index = self.branch_layer_index_for_scale[cur_branch_index]
        return boxes, classes
     
    def predict(self, X):

        boxes_for_scale, classes_for_scale = self.predict_raw(X)
        boxes_list = []
        classes_list = []
        for boxes, classes, aspect_ratios in zip(boxes_for_scale, classes_for_scale, self.aspect_ratios_for_scale):
            batch_size, out_channels, h, w = boxes.size()
            boxes = boxes.view(batch_size, len(aspect_ratios), self.nb_coords, h, w)
            boxes = boxes.transpose(2, 4)
            boxes = boxes.contiguous()
            boxes = boxes.view(-1, self.nb_coords)
            boxes_list.append(boxes)
            
            classes = classes.view(batch_size, len(aspect_ratios), self.nb_classes, h, w) 
            classes = classes.transpose(1, 3)
            classes = classes.contiguous()
            classes = classes.view(-1, self.nb_classes)
            classes = classes.contiguous()
            classes_list.append(classes)
        
        boxes = torch.cat(boxes_list, dim=1)
        classe = torch.cat(classes_list, dim=1)
        return boxes, classes

    def forward(self, X):
        return self.predict_raw(X)
