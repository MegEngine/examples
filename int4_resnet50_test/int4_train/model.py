"""ResNet optimized for quantization, idential after modification."""
import math
from typing import Union

import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import megengine.module.qat as qat
from megengine.quantization.qconfig import QConfig
from megengine.quantization.quantize import quantize, quantize_qat
from functools import partial
from megengine.tensor import Tensor, Parameter
from megengine.core.tensor.dtype import QuantDtypeMeta
from megengine.quantization import fake_quant


class TQT(fake_quant.TQT):
    r"""
    TQT: https://arxiv.org/abs/1903.08066 Trained Quantization Thresholds
    for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks.

    :param dtype: a string or :class:`~.QuantDtypeMeta` indicating the target
        quantization dtype of input.
    :param enable: whether do ``normal_forward`` or ``fake_quant_forward``.
    """

    def __init__(
        self, dtype: Union[str, QuantDtypeMeta], enable: bool = True, **kwargs
    ):
        super().__init__(dtype, enable, **kwargs)
        self.scale = Parameter(1.0, dtype="float32")
        self.zero_point = Tensor(0.0, dtype="float32")

    def get_qparams(self):
        return fake_quant.create_qparams(
            fake_quant.QuantMode.SYMMERTIC,
            self.dtype,
            scale=2 ** self.scale,
            zero_point=self.zero_point,
        )


class Int8ToUint4(M.Module):
    def __init__(self):
        super().__init__()
        self.dequant = M.DequantStub()
        self.quant = M.QuantStub()

    def forward(self, x):
        return self.quant(self.dequant(x))


class BasicBlock(M.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm=M.BatchNorm2d,
    ):
        assert norm is M.BatchNorm2d, "Quant mode only support BatchNorm2d currently."
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv_bn_relu1 = M.ConvBnRelu2d(
            in_channels, channels, 3, stride, padding=dilation, bias=False
        )
        self.conv_bn2 = M.ConvBn2d(channels, channels, 3, 1, padding=1, bias=False)
        self.downsample = (
            M.Identity()
            if in_channels == channels and stride == 1
            else M.ConvBn2d(in_channels, channels, 1, stride, bias=False)
        )
        self.add = M.Elemwise("FUSE_ADD_RELU")

    def forward(self, x):
        identity = x
        x = self.conv_bn_relu1(x)
        x = self.conv_bn2(x)
        identity = self.downsample(identity)
        x = self.add(x, identity)
        return x


class Bottleneck(M.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm=M.BatchNorm2d,
    ):
        assert norm is M.BatchNorm2d, "Quant mode only support BatchNorm2d currently."
        super(Bottleneck, self).__init__()
        width = int(channels * (base_width / 64.0)) * groups
        self.conv_bn_relu1 = M.ConvBnRelu2d(in_channels, width, 1, 1, bias=False)
        self.conv_bn_relu2 = M.ConvBnRelu2d(
            width,
            width,
            3,
            stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.conv_bn3 = M.ConvBn2d(width, channels * self.expansion, 1, 1, bias=False)
        self.downsample = (
            M.Identity()
            if in_channels == channels * self.expansion and stride == 1
            else M.ConvBn2d(
                in_channels, channels * self.expansion, 1, stride, bias=False
            )
        )
        self.add = M.Elemwise("FUSE_ADD_RELU")

    def forward(self, x):
        identity = x
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn3(x)
        identity = self.downsample(identity)
        x = self.add(x, identity)
        return x


class ResNet(M.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm=M.BatchNorm2d,
    ):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.quant = M.QuantStub()
        self.dequant = M.DequantStub()
        self.conv_bn_relu1 = M.ConvBnRelu2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.int8_to_uint4 = Int8ToUint4()
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm=norm)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            norm=norm,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            norm=norm,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            norm=norm,
        )
        self.fc = M.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)
            elif isinstance(m, M.Linear):
                M.init.msra_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    M.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    M.init.zeros_(m.bn2.weight)

    def _make_layer(
        self, block, channels, blocks, stride=1, dilate=False, norm=M.BatchNorm2d
    ):
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(
            block(
                self.in_channels,
                channels,
                stride,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm=norm,
            )
        )
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm=norm,
                )
            )

        return M.Sequential(*layers)

    def extract_features(self, x):
        outputs = {}
        x = self.conv_bn_relu1(x)
        x = self.int8_to_uint4(x)
        x = self.maxpool(x)
        outputs["stem"] = x

        x = self.layer1(x)
        outputs["res2"] = x
        x = self.layer2(x)
        outputs["res3"] = x
        x = self.layer3(x)
        outputs["res4"] = x
        x = self.layer4(x)
        outputs["res5"] = x

        return outputs

    def forward(self, x):
        x = self.quant(x)
        x = self.extract_features(x)["res5"]
        x = self.dequant(x)

        x = F.avg_pool2d(x, 7)
        x = F.flatten(x, 1)
        x = self.fc(x)

        return x


def convert_qat(model):
    qconfig = QConfig(
        weight_observer=None,
        act_observer=None,
        weight_fake_quant=partial(TQT, dtype="qint4"),
        act_fake_quant=partial(TQT, dtype="quint4"),
    )

    model.fc.disable_quantize()
    for module in model.modules():
        if isinstance(module, M.QuantStub):
            module.disable_quantize()

    model_qat = quantize_qat(model, qconfig=qconfig)

    model_qat.conv_bn_relu1.weight_fake_quant = TQT(dtype='qint8')

    for module in model_qat.modules():
        if isinstance(module, (qat.Conv2d, qat.ConvBn2d)):
            module.act_fake_quant = TQT(dtype='qint4')
            module.act_fake_quant.zero_point[...] = 8
        if isinstance(module, qat.Elemwise):
            assert module.method in ['ADD', 'FUSE_ADD_RELU', 'add', 'fuse_add_relu']
            if module.method in ['add', 'ADD']:
                module.act_fake_quant = TQT(dtype='qint4')
                module.act_fake_quant.zero_point[...] = 8
    
    for module in model_qat.modules():
        if isinstance(module, BasicBlock):
            module.conv_bn2.act_fake_quant.disable()
        if isinstance(module, Bottleneck):
            module.conv_bn3.act_fake_quant.disable()
    
    return model_qat


def convert_quantized(model, model_path):
    qconfig = QConfig(
        weight_observer=None,
        act_observer=None,
        weight_fake_quant=partial(TQT, dtype="qint4"),
        act_fake_quant=partial(TQT, dtype="quint4"),
    )

    model.fc.disable_quantize()
    model_qat = quantize_qat(model, qconfig=qconfig)

    model_qat.quant.act_fake_quant = TQT(dtype='qint8')
    model_qat.quant.act_fake_quant.scale[...] = 0

    model_qat.conv_bn_relu1.weight_fake_quant = TQT(dtype='qint8')
    model_qat.conv_bn_relu1.act_fake_quant = TQT(dtype='qint8')

    checkpoint = mge.load(model_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_qat.load_state_dict(state_dict, strict=False)

    model_qat.int8_to_uint4.quant.act_fake_quant.scale[...] = model_qat.conv_bn_relu1.act_fake_quant.scale.numpy()

    model_quantized = quantize(model_qat)
    return model_quantized
    

def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    m = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return m


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    m = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return m
