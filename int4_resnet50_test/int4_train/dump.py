import argparse
import numpy as np
import bisect
import os
import time

import model as resnet_model

import megengine
import megengine.data as data
import megengine.functional as F
import megengine.data.transform as T
import megengine.distributed as dist
from megengine.data.dataset import ImageNet
from megengine.module.qat import QATModule
from megengine import jit
# bias doesn't apply quant. MegEngine will support this function in future.
QATModule.apply_quant_bias = lambda self, b, i, w: b

logging = megengine.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(description="MegEngine ImageNet Dump")
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
        "--save",
        default="resnet50-int4.mge",
        help="path to save output .mge file",
    )

    args = parser.parse_args()

    worker(args)


def worker(args):
    # pylint: disable=too-many-statements
    image = np.ones((64, 3, 224, 224))
    print(image.shape)
    data = megengine.Tensor(image)
    
    # build model
    model = resnet_model.__dict__[args.arch]()
    model = resnet_model.convert_qat(model)

    checkpoint = megengine.load(args.model)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


    model.eval()

    @jit.trace(symbolic=True, capture_as_const=True)
    def infer_func(data, model):
        pred = model(data)
        softmax = F.softmax(pred)
        return softmax 

    output = infer_func(data, model=model)
    infer_func.dump(args.save, arg_names=["data"], output_names=["Softmax"])


if __name__ == "__main__":
    main()
