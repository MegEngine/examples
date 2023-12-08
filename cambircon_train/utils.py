# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2023 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import copy
from functools import partial
from typing import Any, Callable, Dict, Tuple

import megengine as mge
import megengine.module as M
from megengine.core._imperative_rt.core2 import Tensor as RawTensor
from megengine.module import Module
from megengine.tensor import Parameter, Tensor
from megengine.traced_module.pytree import tree_flatten


def _convert_tensor_nchw(x: Tensor, inplace: bool = True):
    if not ((x.ndim == 4 or x.ndim == 5) and x.format == "nhwc"):
        return x

    data = x.numpy()
    if inplace:
        # reset will destroy existed backward grad
        x[...] = Tensor(data, format="nchw")
    else:
        # use mge interface to maintain grad
        x = Tensor(data, format="nchw")
    return x


def convert_module_nchw(module: Module, inplace: bool = True):
    if not inplace:
        module = copy.deepcopy(module)

    for name, param in module.named_tensors():
        _convert_tensor_nchw(param, inplace=True)
    return module


def convert_module_nhwc(module: Module, inplace: bool = True):
    from megengine.amp.convert_format import convert_module_format

    return convert_module_format(module, inplace)


def to_device(param: Tensor, device, inplace: bool = True):
    if inplace:
        param[...] = param.to(device)
    else:
        param = param.to(device)
    return param


def module_to_device(module: Module, device):
    for name, param in module.named_tensors():
        to_device(param, device)
    return module


def tree_map(items, func):
    nodes, treed_def = tree_flatten(
        items, is_const_leaf=lambda node: not isinstance(node, (RawTensor, Parameter, Tensor))
    )
    nodes = [func(node) for node in nodes]
    items = treed_def.unflatten(nodes)
    return items


class NHWCForwardWrapper:
    TO_NHWC = 0x01
    TO_NCHW = 0x02
    ALL = 0x03

    def __init__(self, func: Callable, mode: int = ALL) -> None:
        self.func = func
        self.mode = mode

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.mode & self.TO_NHWC:
            args, kwargs = tree_map((args, kwargs), self._convert_tensor_nhwc_format)
        outputs = self.func(*args, **kwargs)
        if self.mode & self.TO_NCHW:
            outputs = tree_map(outputs, self._convert_tensor_nchw_format)
        return outputs

    @classmethod
    def _convert_tensor_nchw_format(cls, tensor):
        if not ((tensor.ndim == 4) and tensor.format == "nhwc"):
            return tensor
        tensor.format = "nchw"
        return tensor

    @classmethod
    def _convert_tensor_nhwc_format(cls, tensor):
        if not ((tensor.ndim == 4) and tensor.format != "nhwc"):
            return tensor
        tensor.format = "nhwc"
        return tensor


class NHWCModelWrapper:
    NHWC_BASE_MODULE = (
        M.Conv2d.__base__,
        M.ConvTranspose2d.__base__,
        M.AvgPool2d.__base__,
    )
    NHWC_INPUTS_MODULE = (M.Conv2d.__base__,)

    def wrap(self, module: Module, auto: bool = False, inplace: bool = True):
        if not inplace:
            module = copy.deepcopy(module)
        if auto:
            module = self._wrap_submodule(module)
        else:
            module = self._wrap_module(module)
        return module

    def _wrap_module(self, module: Module):
        module = convert_module_nhwc(module=module, inplace=True)
        func = getattr(module, "forward")
        if not isinstance(func, NHWCForwardWrapper):
            setattr(module, "forward", NHWCForwardWrapper(func, NHWCForwardWrapper.ALL))
        for mod in module.modules():
            if not isinstance(mod, self.NHWC_INPUTS_MODULE):
                continue
            func = getattr(mod, "forward")
            if not isinstance(func, NHWCForwardWrapper):
                setattr(mod, "forward", NHWCForwardWrapper(func, NHWCForwardWrapper.TO_NHWC))
        return module

    def _wrap_submodule(self, module: Module):
        submodule_names = self._auto_detect_submodule(module)
        for name in submodule_names:
            self._wrap_module(getattr(module, name))
        return module

    def _auto_detect_submodule(self, module: Module):
        submodule_names = set()
        for name, mod in module.named_modules():
            if not isinstance(mod, self.NHWC_BASE_MODULE):
                continue
            submodule_names.add(str.split(name, ".", 1)[0])
        print(submodule_names)
        return submodule_names

    def unwrap(self, module: Module, inpalce: bool = True):
        if not inpalce:
            module = copy.deepcopy(module)
        module = convert_module_nchw(module)
        self._unwrap_module(module)
        for mod in module.modules():
            self._unwrap_module(mod)
        return module

    def _unwrap_module(self, module: Module):
        func = getattr(module, "forward")
        if not isinstance(func, NHWCForwardWrapper):
            return
        setattr(module, "forward", func.func)
        delattr(module, "forward")


def wrap_nhwc(module: Module, auto: bool = False, inplace: bool = True):
    wrapper = NHWCModelWrapper()
    module = wrapper.wrap(module, auto, inplace)
    return module


def unwrap_nhwc(module: Module, inplace: bool = False):
    wrapper = NHWCModelWrapper()
    module = wrapper.unwrap(module, inplace)
    return module


def is_cambricon() -> bool:
    device = mge.device.what_is_xpu()
    return "cambricon" in device


def _hack_trace_module():
    if not is_cambricon():
        return
    import megengine.traced_module as tm1
    import megengine.traced_module.traced_module as tm2

    origin_trace_module = tm1.trace_module

    def trace_module(module: Module, *args: Tuple[Any], **kwargs: Dict[str, Any]):
        module = unwrap_nhwc(module, inplace=False)
        module = module_to_device(module, "cpu0")
        module.eval()
        args, kwargs = tree_map((args, kwargs), partial(to_device, device="cpu0", inplace=False))
        device = mge.get_default_device()
        mge.set_default_device("cpu0")
        tm_module = origin_trace_module(module, *args, **kwargs)
        mge.set_default_device(device)
        module = module_to_device(module, "xpux")
        return tm_module

    tm1.trace_module = trace_module
    tm2.trace_module = trace_module


_hack_trace_module()
