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

logging = megengine.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(description="MegEngine ImageNet Testing")
    parser.add_argument("-d", "--data", metavar="DIR", help="path to imagenet dataset")
    parser.add_argument(
        "-m", "--model", metavar="PKL", default=None, help="path to model checkpoint"
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

def worker(args):
    # performance test
    os.system("python3 -m megengine.tools.load_network_and_run {} --fast-run --iter 60 --profile mma.json --weight-preprocess".format(args.model))
    # build dataset
    valid_dataloader = build_dataset(args)

    # load model
    from megengine.core.tensor import megbrain_graph as G
    from megengine.utils.comp_graph_tools import GraphInference
    model = GraphInference(args.model)
    

    def valid_step(image, label):
        logits = model.run(image)['Softmax']
        from megengine import Tensor
        logits = Tensor(logits)
        loss = F.nn.cross_entropy(logits, label)
        #import pdb
        #pdb.set_trace()
        acc1, acc5 = F.topk_accuracy(logits, label, topk=(1, 5))
        return loss, acc1, acc5

    _, valid_acc1, valid_acc5 = valid(valid_step, valid_dataloader, args)
    logging.info(
        "Test Acc@1 %.3f, Acc@5 %.3f",
        valid_acc1,
        valid_acc5,
    )

    os.system("python3 -m megengine.tools.profile_analyze mma.json")

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
            dataset.append(np.array(img))

    dataset = np.array(dataset)
    labelset = np.load(args.data + "label.npy")
    valid_dataset = ArrayDataset(dataset, labelset)
    
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=100, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
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
