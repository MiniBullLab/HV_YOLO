from easyAI.base_block.utility_layer import RouteLayer, Upsample
from easyAI.base_block.utility_block import ConvBNActivationBlock
from easyAI.base_block.utility_block import ConvActivationBlock
from easyAI.base_name.block_name import BatchNormType, ActivationType, BlockType
from easyAI.base_name.loss_name import LossType
from easyAI.loss.cross_entropy2d import CrossEntropy2d
from easyAI.base_model.mobilenetv2 import MobileNetV2
from easyAI.base_model.base_model import *
from easyAI.base_name.model_name import ModelName


class MobileV2FCN(BaseModel):

    def __init__(self, class_num=2):
        super().__init__()
        self.set_name(ModelName.MobileV2FCN)
        self.class_number = class_num
        self.bn_name = BatchNormType.BatchNormalize
        self.activation_name = ActivationType.ReLU6
        self.lossList = []
        self.out_channels = []
        self.index = 0
        self.create_model()

    def create_model(self):
        basic_model = MobileNetV2(bnName=self.bn_name, activationName=self.activation_name)
        base_out_channels = basic_model.getOutChannelList()
        self.add_block_list(BlockType.BaseNet, basic_model, base_out_channels[-1])

        input_channel = self.out_channels[-1]
        output_channel = base_out_channels[-1] // 2
        self.make_block(input_channel, output_channel)

        layer1 = Upsample(scale_factor=2, mode='bilinear')
        self.add_block_list(layer1.get_name(), layer1, self.out_channels[-1])

        layer2 = RouteLayer('13')
        output_channel = sum([base_out_channels[i] if i >= 0 else self.out_channels[i] for i in layer2.layers])
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        input_channel = self.out_channels[-1]
        output_channel = base_out_channels[-1] // 2
        conv1 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=1, stride=1, padding=0,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, output_channel)

        layer3 = RouteLayer('-1,-3')
        output_channel = sum([base_out_channels[i] if i >= 0 else self.out_channels[i] for i in layer3.layers])
        self.add_block_list(layer3.get_name(), layer3, output_channel)

        layer4 = Upsample(scale_factor=2, mode='bilinear')
        self.add_block_list(layer4.get_name(), layer4, self.out_channels[-1])

        input_channel = self.out_channels[-1]
        output_channel = base_out_channels[-1] // 4
        self.make_block(input_channel, output_channel)

        layer5 = RouteLayer('6')
        output_channel = sum([base_out_channels[i] if i >= 0 else self.out_channels[i] for i in layer5.layers])
        self.add_block_list(layer5.get_name(), layer5, output_channel)

        input_channel = self.out_channels[-1]
        output_channel = base_out_channels[-1] // 4
        conv2 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=1, stride=1, padding=0,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, output_channel)

        layer6 = RouteLayer('-1,-3')
        output_channel = sum([base_out_channels[i] if i >= 0 else self.out_channels[i] for i in layer6.layers])
        self.add_block_list(layer6.get_name(), layer6, output_channel)

        input_channel = self.out_channels[-1]
        output_channel = base_out_channels[-1] // 8
        self.make_block(input_channel, output_channel)

        input_channel = self.out_channels[-1]
        output_channel = self.class_number
        conv3 = ConvActivationBlock(input_channel, output_channel,
                                    kernel_size=1, stride=1, padding=0,
                                    activationName=ActivationType.Linear)
        self.add_block_list(conv3.get_name(), conv3, output_channel)

        layer7 = Upsample(scale_factor=8, mode='bilinear')
        self.add_block_list(layer1.get_name(), layer7, self.out_channels[-1])

        loss = CrossEntropy2d(ignore_index=250, size_average=True)
        self.add_block_list(LossType.CrossEntropy2d, loss, self.out_channels[-1])
        self.lossList.append(loss)

    def make_block(self, input_channel, output_channel):
        conv1 = ConvBNActivationBlock(input_channel, output_channel,
                                      kernel_size=1, stride=1, padding=0,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, output_channel)

        temp_input_channel = self.out_channels[-1]
        temp_output_channel = output_channel * 2
        conv2 = ConvBNActivationBlock(temp_input_channel, temp_output_channel,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, temp_output_channel)

        temp_input_channel = self.out_channels[-1]
        temp_output_channel = output_channel
        conv3 = ConvBNActivationBlock(temp_input_channel, temp_output_channel,
                                      kernel_size=1, stride=1, padding=0,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, temp_output_channel)

    def add_block_list(self, block_name, block, output_channel):
        block_name = "%s_%d" % (block_name, self.index)
        self.add_module(block_name, block)
        self.index += 1
        self.out_channels.append(output_channel)

    def freeze_bn(self, freezeBn):
        for m in self.modules():
            if freezeBn and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif BlockType.Convolutional in key:
                x = block(x)
            elif BlockType.ConvActivationBlock in key:
                x = block(x)
            elif BlockType.ConvBNActivationBlock in key:
                x = block(x)
            elif BlockType.Upsample in key:
                x = block(x)
            elif BlockType.MyMaxPool2d in key:
                x = block(x)
            elif BlockType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif BlockType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif BlockType.GlobalAvgPool in key:
                x = block(x)
            elif BlockType.FcLayer in key:
                x = block(x)
            elif LossType.YoloLoss in key:
                output.append(x)
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            layer_outputs.append(x)
        return output
