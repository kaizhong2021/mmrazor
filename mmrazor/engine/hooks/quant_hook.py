# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from typing import Optional, Sequence, Union

from mmengine.dist import master_only
from mmengine.fileio import FileClient, dump
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]

from typing import Dict, List, Union

from mmrazor.fx import CustomTracer, custom_trace
from mmrazor.models import FXModelWrapper


class CustomGraphModuleConverter:

    def __init__(self, model):

        assert isinstance(
            model, FXModelWrapper
        ), f'model must be a `FXModelWrapper`, but got `{type(model)}`'
        self.model = model
        self.tracer = CustomTracer(
            customed_skipped_method=model.customed_skipped_method)

        self.graph_module = custom_trace(
            self.model, customed_skipped_method=model.customed_skipped_method)

    def recompile(self):
        new_graph = self.tracer.trace(self.model)
        self.graph_module.graph = new_graph
        self.graph_module.recompile()

    def tensor_mode(self):
        if self.model.mode == 'tensor':
            return self.graph_module
        self.model.mode = 'tensor'
        self.recompile()
        return self.graph_module

    def loss_mode(self):
        if self.model.mode == 'loss':
            return self.graph_module
        self.model.mode = 'loss'
        self.recompile()
        return self.graph_module

    def predict_mode(self):
        if self.model.mode == 'predict':
            return self.graph_module
        self.model.mode = 'predict'
        self.recompile()
        return self.graph_module


@HOOKS.register_module()
class QuantitiveHook(Hook):
    converter = None
    model = None
    priority = 'NORMAL'

    def before_run(self, runner) -> None:
        self.model = runner.model.module if runner.distributed else runner.model
        self.converter = CustomGraphModuleConverter(self.model)

    def before_train_epoch(self, runner) -> None:
        if runner.distributed:
            runner.model.module = self.converter.loss_mode()
        else:
            runner.model = self.converter.loss_mode()

    def before_test_epoch(self, runner) -> None:
        if runner.distributed:
            runner.model.module = self.converter.predict_mode()
        else:
            runner.model = self.converter.predict_mode()

    def before_val_epoch(self, runner) -> None:
        if runner.distributed:
            runner.model.module = self.converter.predict_mode()
        else:
            runner.model = self.converter.predict_mode()
