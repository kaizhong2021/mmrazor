# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

from mmengine.runner import EpochBasedTrainLoop, IterBasedTrainLoop
from torch.utils.data import DataLoader

from mmrazor.models import CustomTracer, FXModelWrapper, custom_trace
from mmrazor.registry import LOOPS


class CustomGraphModuleConverter:

    def __init__(self, model):

        assert isinstance(
            model, FXModelWrapper
        ), f'model must be a `FXModelWrapper`, but got {type(model)}'
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
        self.model.mode = 'tensor'
        self.recompile()
        return self.graph_module

    def loss_mode(self):
        self.model.mode = 'loss'
        self.recompile()
        return self.graph_module

    def predict_mode(self):
        self.model.mode = 'predict'
        self.recompile()
        return self.graph_module


@LOOPS.register_module()
class QuantTrainLoop(EpochBasedTrainLoop):

    def __init__(self,
                 runner,
                 dataloader: Union[Dict, DataLoader],
                 max_epochs: int,
                 val_begin: int = 1,
                 val_interval: int = 1) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval)

        self.model = self._runner.model
        self.converter = CustomGraphModuleConverter(self.model)
