from typing import *

from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from .. import models
import os
import json

from config.settings import settings
from logger_config import logger


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = None, revision: str = None, model_revisions: dict = None) -> "Pipeline":
        """
        Load a pretrained model.
        
        Args:
            path (str): Path to the model.
            config_file (str, optional): Path to config file. If None, uses settings.pipeline_config_path.
            revision (str, optional): Specific revision/commit to use for the main model.
            model_revisions (dict, optional): Dict mapping model IDs to specific revisions.
        """
        
        if config_file is None:
            config_file = settings.pipeline_config_path
    
        is_local = os.path.exists(config_file)
    
        if is_local:
            config_file_path = config_file
        else:
            config_file_path = hf_hub_download(path, "pipeline.json", revision=revision)

        with open(config_file_path, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        model_revisions = model_revisions or {}
        
        logger.debug(f"Loading pipeline from {path} with revision={revision}")
        logger.debug(f"Model revisions mapping: {model_revisions}")
        
        for k, v in args['models'].items():
            model_revision = model_revisions.get(v, revision if f"{path}/{v}" == v else None)
            
            logger.debug(f"Loading model '{k}' from '{v}' with revision={model_revision}")
            
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}", revision=model_revision)
                logger.debug(f"Successfully loaded model '{k}' from {path}/{v}")
            except Exception as e:
                logger.debug(f"Failed to load from {path}/{v}, trying {v} directly")
                _models[k] = models.from_pretrained(v, revision=model_revisions.get(v))
                logger.debug(f"Successfully loaded model '{k}' from {v}")

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))