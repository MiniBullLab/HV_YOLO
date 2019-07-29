import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from base_model.baseModel import *
from utils.parse_config import *
from base_model.baseLayer import EmptyLayer, Upsample
from base_model.mobilenetv2 import MobileNetV2

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    # create basic model
    basicModel = MobileNetV2(width_mult=1.0)
    out_channels = basicModel.out_channels

    hyperparams = module_defs.pop(0)
    output_filters = [out_channels[-1]]
    module_list = nn.ModuleList()

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'relu':
                modules.add_module('leaky_%d' % i, nn.ReLU())

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']), mode='bilinear')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([out_channels[i] if i >= 0 else output_filters[i] for i in layers])
            # filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class MobileFCN(BaseModel):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=[224, 224], freeze_bn=False):
        super().__init__()
        self.setModelName("MobileFCN")
        self.module_defs = parse_model_config(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['width'] = img_size[0]
        self.module_defs[0]['height'] = img_size[1]

        self.basicModel = MobileNetV2(width_mult=1.0)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size

        self._init_weight()

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _init_weight(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        layer_outputs = []

        blocks = self.basicModel(x)
        x = blocks[-1]

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] if i < 0 else blocks[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            layer_outputs.append(x)

        return x