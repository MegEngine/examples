import argparse
import time
import os
import numpy as np

# pylint: disable=import-error
import model as resnet_model

import megengine
import megengine.data as data
import megengine.data.transform as T
import megengine.functional as F
from megengine.data.dataset import ImageNet

import os
import argparse
import time
from collections import OrderedDict
from IPython import embed
import numpy as np
from tqdm import tqdm

import megengine as mge
import megengine.module as M
import megengine.functional as F
from megengine.quantization.fake_quant import TQT
from megengine.module.qat import QATModule
from megengine.quantization.quantize import disable_fake_quant, enable_fake_quant
from functools import partial
from collections import OrderedDict

QATModule.apply_quant_bias = lambda self, b, i, w: b

logging = megengine.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(description="MegEngine ImageNet Training")
    parser.add_argument("-d", "--data", metavar="DIR", help="path to imagenet dataset")
    parser.add_argument(
        "-a",
        "--arch",
        default="resnet50",
        help="model architecture (default: resnet50)",
    )
    parser.add_argument(
        "-m", "--model", metavar="PKL", default=None, help="path to model checkpoint"
    )
    parser.add_argument(
        "-o", "--output_path", metavar="PKL", default='log/model_qat_init.pkl', help="path to model checkpoint"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=200,
        type=int,
        help="batch size for calibrate (default: 200)",
    )

    parser.add_argument("-j", "--workers", default=2, type=int)

    args = parser.parse_args()

    calib_data = prepare_calibrate_data(args)

    model = resnet_model.__dict__[args.arch]()
    model_qat = resnet_model.convert_qat(model)
    checkpoint = mge.load(args.model)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_qat.load_state_dict(state_dict, strict=False)

    model_qat = init_scale_offline(model_qat, calib_data, block_map=None, weight=True, act=True)
    model_qat = update_bn(args, model_qat)
    mge.save({"state_dict": model_qat.state_dict()}, args.output_path)
    from megengine.core._imperative_rt.core2 import close as mge_close
    mge_close()

def update_bn(args, model_qat):
    dataset = ImageNet(args.data, train=True)
    sampler = data.Infinite(
        data.RandomSampler(dataset, batch_size=32, drop_last=True)
    )
    dataloader = data.DataLoader(
        dataset,
        sampler=sampler,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[128.0, 128.0, 128.0], std=[1.0, 1.0, 1.0]
                ),  # BGR
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )

    data_queue = iter(dataloader)

    model_qat.train()
    for _ in range(100):
        image, _ = next(data_queue)
        image = megengine.tensor(image, dtype="float32")
        model_qat(image)
    
    return model_qat


def prepare_calibrate_data(args):
    
    if os.path.exists('log/images.npy'):
        cali_data = np.load('log/images.npy')
        return cali_data
    
    else:
        dataset = ImageNet(args.data, train=True)
        sampler = data.Infinite(
            data.RandomSampler(dataset, batch_size=args.batch_size, drop_last=True)
        )
        dataloader = data.DataLoader(
            dataset,
            sampler=sampler,
            transform=T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.Normalize(
                        mean=[128.0, 128.0, 128.0], std=[1.0, 1.0, 1.0]
                    ),  # BGR
                    T.ToMode("CHW"),
                ]
            ),
            num_workers=args.workers,
        )

        data_queue = iter(dataloader)
        images, _ = next(data_queue)
        np.save('log/images.npy', images)
        return images


def search_scale_by_mse(inp: mge.Tensor, qmax: int, qmin: int):
    inp_max = F.abs(inp).max()
    best_score = 1e+10
    best_scale = None
    for i in range(80):
        new_max = inp_max * (1.0 - (i * 0.01))
        if qmin == 0:
            scale = new_max / qmax
        else:
            scale = 2 * new_max / (qmax - qmin)
        inp_int = F.round(inp / scale)
        inp_quant = F.clip(inp_int, qmin, qmax)
        inp_qfloat = inp_quant * scale
        score = F.pow(F.abs(inp - inp_qfloat), 2.4).mean()
        if score < best_score:
            best_score = score
            best_scale = scale
    return best_scale


def search_scale_numpy(tensor, qmax, qmin):
    old_scale = np.max(np.abs(tensor)) / qmax
    new_scale = old_scale
    for _ in tqdm(range(40)):
        qtensor = np.clip(np.round(tensor / old_scale), qmin, qmax)
        new_scale = np.dot(tensor, qtensor) / np.dot(qtensor, qtensor)
        if abs(old_scale - new_scale) < 1e-4:
            break
        old_scale = new_scale
    scale = new_scale
    return scale


def get_scale(key, qmax, qmin):
    with open('tmp/{}.bin'.format(key), 'rb') as f:
        feature = np.array(np.fromfile(f, dtype='float16'), dtype='float32')
        best_scale = search_scale_numpy(feature, qmax, qmin)
        return key, best_scale


def init_scale_offline(module: M.Module,
                       calib_data: np.array,
                       block_map: list = [QATModule],
                       weight=True,
                       act=True,
                       ):
    def init_quant_modules():
        from collections import OrderedDict
        quant_modules = OrderedDict()

        def check_type(target, name):
            for m in block_map:
                if isinstance(m, str):
                    if name == m:
                        return True
                    continue
                if isinstance(target, m):
                    return True
            return False

        def fill_quant_modules(mod, inputs, outputs, name):
            quant_modules[name] = mod
            mod._forward_hooks.clear()
            return None

        def set_hook(mod: M.Module, prefix: str = None):
            _prefix = "" if prefix is None else prefix + "."
            for key, submodule in list(mod._flatten(with_key=True,
                                                    recursive=False, predicate=lambda m: isinstance(m, M.Module))):
                submodule_name = '{}{}'.format(_prefix, key)
                if check_type(submodule, submodule_name):
                    submodule.register_forward_hook(partial(fill_quant_modules, name=submodule_name))
                    continue
                set_hook(submodule, submodule_name)

        module.eval()
        set_hook(module)
        disable_fake_quant(module)
        module(mge.Tensor(calib_data[:1], dtype='float32'))
        return quant_modules

    def disable_basic_block_conv2_fake_quant(model):
        for mod in model.modules():
            if isinstance(mod, resnet_model.BasicBlock):
                mod.conv_bn2.act_fake_quant.disable()
            if isinstance(mod, resnet_model.Bottleneck):
                mod.conv_bn3.act_fake_quant.disable()

    def init_weight_scale(quant_modules_with_weight: OrderedDict):
        module.eval()

        def find_weight_scale(mod, inputs, outputs):
            best_scale = search_scale_by_mse(inputs[0], mod.qmax, mod.qmin)
            mod.scale[...] = F.log(best_scale) / F.log(2)

        disable_fake_quant(module)

        for key, mod in quant_modules_with_weight.items():
            mod.weight_fake_quant.register_forward_hook(find_weight_scale)

        module(mge.Tensor(calib_data[:1], dtype='float32'))

        for key, mod in quant_modules_with_weight.items():
            mod.weight_fake_quant._forward_hooks.clear()

        enable_fake_quant(module)
        disable_basic_block_conv2_fake_quant(module)

    def init_act_scale(quant_modules_with_act: OrderedDict,
                       batch_size: int = 32):
        num_sample = calib_data.shape[0]
        module.eval()

        os.makedirs('tmp', exist_ok=True)
        feature_file_dict = {}
        for key in quant_modules_with_act:
            feature_file_dict[key] = open('tmp/{}.bin'.format(key), 'wb')

        def save(name):
            def compute_feature_map(mod, inputs, outputs):
                content = inputs[0].numpy()
                content = content.astype('float16').tobytes()
                feature_file_dict[name].write(content)

            return compute_feature_map

        disable_fake_quant(module)

        for key, mod in quant_modules_with_act.items():
            mod.act_fake_quant.register_forward_hook(save(key))

        for i in tqdm(range(0, num_sample, batch_size)):
            data = calib_data[i: i + batch_size]
            module(mge.Tensor(data, dtype='float32'))

        for key, mod in quant_modules_with_act.items():
            mod.act_fake_quant._forward_hooks.clear()

        for key, file in feature_file_dict.items():
            file.close()

        for key in quant_modules_with_act:
            mod = quant_modules_with_act[key]
            qmax = mod.act_fake_quant.qmax
            qmin = mod.act_fake_quant.qmin
            _, best_scale = get_scale(key, qmax, qmin)
            mod.act_fake_quant.scale[...] = F.log(mge.Tensor(best_scale, dtype=np.float32)) / F.log(2.0)

        enable_fake_quant(module)
        disable_basic_block_conv2_fake_quant(module)

    if block_map is None:
        block_map = []
    block_map.append(QATModule)
    quant_modules = init_quant_modules()
    quant_modules_with_weight = OrderedDict()
    quant_modules_with_act = OrderedDict()
    for key, quant_module in quant_modules.items():
        if isinstance(quant_module.act_fake_quant, TQT):
            quant_modules_with_act[key] = quant_module
        if isinstance(quant_module.weight_fake_quant, TQT):
            quant_modules_with_weight[key] = quant_module

    if weight:
        init_weight_scale(quant_modules_with_weight)
    if act:
        init_act_scale(quant_modules_with_act)

    return module


if __name__ == "__main__":
    main()
