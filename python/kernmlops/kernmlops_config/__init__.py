import dataclasses
from pathlib import Path
from typing import Any, Mapping, MutableMapping

DEFAULT_CONFIG_FILE = Path("defaults.yaml")


@dataclasses.dataclass(frozen=True)
class ConfigBase:

    def merge(self, config_overrides: Mapping[str, Any]) -> "ConfigBase":
        def _merge(old: MutableMapping[str, Any], new: Mapping[str, Any]) -> None:
            for k, v in new.items():
                # if k is not specified in the old mapping, add it
                if k not in old:
                    old[k] = v
                # if the old value is a ConfigBase, call merge
                elif isinstance(old[k], ConfigBase):
                    old[k] = old[k].merge(v)
                # if either mappings' value is not a dict take the new value
                elif not (isinstance(v, dict) and isinstance(old[k], dict)):
                    old[k] = v
                # if both mappings' values are dicts merge them
                else:
                    _merge(old=old[k], new=v)

        merged_config = dict[str, Any]()
        for field in dataclasses.fields(self):
            merged_config[field.name] = getattr(self, field.name)
        _merge(old=merged_config, new=config_overrides)
        return dataclasses.replace(self, **merged_config)


__all__ = [
    "ConfigBase",
    "DEFAULT_CONFIG_FILE",
]
