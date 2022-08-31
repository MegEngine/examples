import argparse
import time
import os

import megengine
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
from matplotlib import pyplot as plt

import numpy as np
import cv2
from megengine.data import DataLoader
from megengine.data.dataset import ArrayDataset

from megengine.utils.network import Network as Net
from megengine.functional.external import tensorrt_runtime_opr
import io

from megengine.core.tensor import megbrain_graph as G
from megengine.utils.comp_graph_tools import GraphInference

logging = megengine.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(description="MegEngine ImageNet Testing")
    parser.add_argument("-d", "--data", metavar="DIR", help="path to imagenet dataset")
    parser.add_argument(
        "-m", "--model", metavar="engine", default=None, help="path to trt engine"
    )

    parser.add_argument(
        "-p",
        "--print-freq",
        default=20,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    args = parser.parse_args()
    worker(args)

def test_tensorrt(model_file):
    with open(model_file, "rb") as f:
        data = f.read()
        net = Net()
        a = net.make_input_node(shape=(256, 3, 224, 224), dtype=np.float32, name="input")
        out = tensorrt_runtime_opr([a], data=data)
        nodes = net.add_dep_oprs(*out)
        net.add_output(*nodes)

        dump_file = io.BytesIO()
        net.dump(dump_file)
        dump_file.seek(0)

        graph = GraphInference(dump_file)
        return graph

def worker(args):
    # performance test
    os.system("/usr/local/cuda-11.4-cudnn-8.2.1-trt-7.2.2.3-libs/TensorRT-7.2.2.3/bin/trtexec  --loadEngine={} --optShapes=input:256x3x224x224 --int8".format(args.model))
    # build dataset
    valid_dataloader = build_dataset(args)

    # load model
    model = test_tensorrt(args.model)

    def valid_step(image, label):
        logits = model.run(image)['']
        from megengine import Tensor
        logits = Tensor(logits)
        loss = F.nn.cross_entropy(logits, label)
        acc1, acc5 = F.topk_accuracy(logits, label, topk=(1, 5))
        return loss, acc1, acc5

    _, valid_acc1, valid_acc5 = valid(valid_step, valid_dataloader, args)
    logging.info(
        "Test Acc@1 %.3f, Acc@5 %.3f",
        valid_acc1,
        valid_acc5,
    )


def valid(func, data_queue, args):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    clck = AverageMeter("Time")

    t = time.time()
    for step, (image, label) in enumerate(data_queue):
        image = megengine.tensor(image, dtype="float32")
        label = megengine.tensor(label, dtype="int32")

        n = image.shape[0]

        loss, acc1, acc5 = func(image, label)

        objs.update(loss.item(), n)
        top1.update(100 * acc1.item(), n)
        top5.update(100 * acc5.item(), n)
        clck.update(time.time() - t, n)
        t = time.time()

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info("Test step %d, %s %s %s %s", step, objs, top1, top5, clck)

    return objs.avg, top1.avg, top5.avg


def build_dataset(args):
    dataset = []
    img_dir = args.data + "img/"
    
    for _,_,k in os.walk(img_dir):
        k.sort()
        for k0 in k:
            img = cv2.imread(img_dir + k0)
            dataset.append(np.array(img)[...,::-1])

    dataset = np.array(dataset)
    labelset = np.load(args.data + "label.npy")
    valid_dataset = ArrayDataset(dataset, labelset)
    
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=256, drop_last=True
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
                ),
                T.ToMode("CHW"),
            ]
        ),
        num_workers=4,
    )
    return valid_dataloader


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":.3f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main()
