import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from prettytable import PrettyTable
import math
import logging

codes_field = 'codes'
pooled_codes_field = 'pooled_codes'
classes_field = 'categories'


class CustomNet(nn.Module):
    def __init__(self, model_name, custom_trained_model=True):
        super().__init__()

        # define standard architecture
        specific_model = getattr(models, model_name)
        if custom_trained_model:
            self.model = specific_model(pretrained=False)
        else:
            self.model = specific_model(pretrained=True)

    def _find_number_of_features(self):

        n_features = None

        modules = list(self.model.named_modules())
        for n in reversed(range(len(modules))):

            if hasattr(modules[n][1], "out_channels"):
                n_features = modules[n][1].out_channels
                logging.info("{} features were discovered as out_channels in CNN layer".format(n_features))
                break

        if n_features is None:
            logging.info("number of features cannot be found")
            raise

        return n_features

    def add_classifier(self, n_features=None, n_hidden=64, *, n_classes):

        if not n_features:
            self.n_features = self._find_number_of_features()
        else:
            self.n_features = n_features
            logging.info("{} features were requested".format(self.n_features))

        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.classifier = nn.Sequential(
            nn.Linear(self.n_features * 1 * 1, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_classes),

        )

    def req_grad(self, n, status):
        '''
        Sets requires_grad field for network parameters up to the features convolutional layer 'n' to freeze (status=False) or unfreeze (status=True) the weights te be learned
        :param n: the number of first feature convolutional layers to be affected [0, n-1]
        :param status: False - freeze weights, True - unfreeze weights
        :return:
        '''
        for layer_n in range(n):
            for parameter in self.model.features[layer_n].parameters():
                parameter.requires_grad = status

    def fv_length(self):
        return self.n_features

    def custom_init_fcl_weights_vgg(self, n_first=0, n_last=None):
        """
        Initializes the weights of fully connected layers according to the VGG recommendations (exists in the original pytorch vgg code)
        :param n_first: first layer to be initialized
        :param n_last: last layer to be initialized
        :return:
        """

        if n_last is None:
            n_last = len(self.model.classifier) - 1

        for layer_n in range(n_first, n_last + 1):
            layer = self.model.classifier[layer_n]

            if isinstance(layer, nn.Linear):
                assert layer.weight.requires_grad is True
                nn.init.normal_(layer.weight, 0, 0.01)  # 0.01
                assert layer.bias.requires_grad is True
                nn.init.constant_(layer.bias, 0)

    def custom_init_fcl_efficientnet(self, n_first=0, n_last=None):

        if n_last is None:
            n_last = len(self.model.classifier) - 1

        for layer_n in range(n_first, n_last + 1):
            layer = self.model.classifier[layer_n]

            if isinstance(layer, nn.Linear):

                init_range = 1.0 / math.sqrt(layer.out_features)
                assert layer.weight.requires_grad is True
                nn.init.uniform_(layer.weight, -init_range, init_range)
                assert layer.bias.requires_grad is True
                nn.init.zeros_(layer.bias)

    def count_parameters(self, verbose=False):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue

            param = parameter.numel()
            table.add_row([name, param])
            total_params += param

        if verbose:
            print(table)
            print(f"Total Trainable Params: {total_params}")

        return total_params

class VGG_11(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'vgg11'

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.add_classifier(n_classes=n_classes)

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class VGG_16(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'vgg16'

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.add_classifier(n_classes=n_classes)

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class VGG_19(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'vgg19'

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.add_classifier(n_classes=n_classes)

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class EfficientNet_B0(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'efficientnet_b0'

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.add_classifier(n_classes=n_classes)

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class EfficientNet_B2(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'efficientnet_b2'

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.add_classifier(n_classes=n_classes)

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class EfficientNet_B0_320(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'efficientnet_b0'
        n_features = 320

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.model.features = self.model.features[:-1]
        self.model.features.add_module('addedBN', nn.BatchNorm2d(n_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.model.features.add_module('AddedSiLu', nn.SiLU(inplace=True))

        self.add_classifier(n_classes=n_classes)
        assert n_features == self.n_features

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}

class EfficientNet_B2_352(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'efficientnet_b2'
        n_features = 352

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.model.features = self.model.features[:-1]
        self.model.features.add_module('addedBN', nn.BatchNorm2d(n_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.model.features.add_module('AddedSiLu', nn.SiLU(inplace=True))

        self.add_classifier(n_classes=n_classes)
        assert n_features == self.n_features

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class ConvNeXt(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'convnext_tiny'

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.add_classifier(n_classes=n_classes)

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class ResNet_18(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'resnet18'

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.add_classifier(n_classes=n_classes)

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        codes = self.model.layer4(x)
        pooled_codes = self.model.avgpool(codes)

        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class DenseNet_121(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'densenet121'
        n_features = 1024

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.add_classifier(n_classes=n_classes, n_features=n_features)

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        codes = F.relu(codes, inplace=True)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class DenseNet_121_512(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'densenet121'
        n_features = 512

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.model.features = self.model.features[:-2]
        self.model.features.add_module('addedBN', nn.BatchNorm2d(n_features))

        self.add_classifier(n_classes=n_classes)
        assert n_features == self.n_features

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        codes = self.model.features(x)
        codes = F.relu(codes, inplace=True)
        pooled_codes = self.model.avgpool(codes)
        pooled_codes = torch.flatten(pooled_codes, 1)
        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}


class VT_B_32(CustomNet):
    def __init__(self, path_trained_model='', *, n_classes, dev):

        model_name = 'vit_b_32'

        super().__init__(model_name=model_name, custom_trained_model=bool(path_trained_model))

        self.add_classifier(n_classes=n_classes)

        # initialize weights of the model
        if path_trained_model:
            self.load_state_dict(torch.load(path_trained_model, map_location=dev))
        else:
            self.custom_init_fcl_weights_vgg()

    def forward(self, x):

        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        pooled_codes = x[:, 0]

        x = self.model.classifier(pooled_codes)

        return {classes_field: x, pooled_codes_field: pooled_codes}





















