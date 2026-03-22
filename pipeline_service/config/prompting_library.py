import json
from os import PathLike
from pathlib import Path
from typing import Dict

import yaml
from pydantic import RootModel

from modules.image_edit.prompting import TextPrompting


class PromptingLibrary(RootModel):
    root: Dict[str, TextPrompting]

    @property
    def promptings(self):
        return self.root

    @classmethod
    def from_file(cls, path: PathLike):
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        return cls.model_validate(data)