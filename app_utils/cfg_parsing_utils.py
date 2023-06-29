import dataclasses
from functools import partial
from typing import Any, Optional, List, Tuple, Type, Union

from h2o_wave import Q, ui

from app_utils.config import default_cfg
from app_utils.utils import get_dataset, make_label
from llm_studio.src import possible_values
from llm_studio.src.utils.config_utils import _get_type_annotation_error
from llm_studio.src.utils.data_utils import read_dataframe
from llm_studio.src.utils.type_annotations import KNOWN_TYPE_ANNOTATIONS


def get_ui_elements(
        cfg: Any,
        q: Q,
        limit: Optional[List[str]] = None,
        pre: str = "experiment/start",
) -> List:
    """For a given configuration setting return the according ui components.

    Args:
        cfg: configuration settings
        q: Q
        limit: optional list of keys to limit
        pre: prefix for client keys

    Returns:
        List of ui elements
    """
    items = []

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()

    cfg_dict = {key: cfg_dict[key] for key in cfg._get_order()}

    for config_item_name, config_item_value in cfg_dict.items():
        if config_item_name.startswith("_") or cfg._get_visibility(config_item_name) < 0:
            if q.client[f"{pre}/cfg_mode/from_cfg"]:
                q.client[f"{pre}/cfg/{config_item_name}"] = config_item_value
            continue
        else:
            type_annotation = type_annotations[config_item_name]
            poss_values, config_item_value = cfg._get_possible_values(
                field=config_item_name,
                value=config_item_value,
                type_annotation=type_annotation,
                mode=q.client[f"{pre}/cfg_mode/mode"],
                dataset_fn=partial(get_dataset, q=q, limit=limit, pre=pre),
            )

            if config_item_name in default_cfg.dataset_keys:
                # reading dataframe
                if config_item_name == "train_dataframe" and (config_item_value != ""):
                    q.client[f"{pre}/cfg/dataframe"] = read_dataframe(config_item_value, meta_only=True)
                q.client[f"{pre}/cfg/{config_item_name}"] = config_item_value
            elif config_item_name in default_cfg.dataset_extra_keys:
                _, config_item_value = get_dataset(config_item_name, config_item_value, q=q, limit=limit, pre=pre)
                q.client[f"{pre}/cfg/{config_item_name}"] = config_item_value
            elif q.client[f"{pre}/cfg_mode/from_cfg"]:
                q.client[f"{pre}/cfg/{config_item_name}"] = config_item_value
        # Overwrite current default values with user_settings
        if q.client[f"{pre}/cfg_mode/from_default"] and f"default_{config_item_name}" in q.client:
            q.client[f"{pre}/cfg/{config_item_name}"] = q.client[f"default_{config_item_name}"]

        if not (_check_dependencies(cfg=cfg, pre=pre, k=config_item_name, q=q)):
            continue

        if (not _is_visible(k=config_item_name, cfg=cfg, q=q)) and q.client[f"{pre}/cfg_mode/from_cfg"]:
            q.client[f"{pre}/cfg/{config_item_name}"] = config_item_value
            continue

        tooltip = cfg._get_tooltips(config_item_name)

        trigger = False
        q.client[f"{pre}/trigger_ks"] = ["train_dataframe"]
        q.client[f"{pre}/trigger_ks"] += cfg._get_nesting_triggers()
        if config_item_name in q.client[f"{pre}/trigger_ks"]:
            trigger = True

        if type_annotation in KNOWN_TYPE_ANNOTATIONS:
            if limit is not None and config_item_name not in limit:
                continue

            items_list = _get_ui_element(
                config_item_name=config_item_name,
                config_item_value=config_item_value,
                poss_values=poss_values,
                type_annotation=type_annotation,
                tooltip=tooltip,
                password="api" in config_item_name,
                trigger=trigger,
                q=q,
                pre=f"{pre}/cfg/",
            )
        elif dataclasses.is_dataclass(config_item_value):
            if limit is not None and config_item_name in limit:
                elements_group = get_ui_elements(cfg=config_item_value, q=q, limit=None, pre=pre)
            else:
                elements_group = get_ui_elements(cfg=config_item_value, q=q, limit=limit, pre=pre)

            if config_item_name == "dataset" and pre != "experiment/start":
                # get all the datasets available
                df_datasets = q.client.app_db.get_datasets_df()
                if not q.client[f"{pre}/dataset"]:
                    if len(df_datasets) >= 1:
                        q.client[f"{pre}/dataset"] = str(df_datasets["id"].iloc[-1])
                    else:
                        q.client[f"{pre}/dataset"] = "1"

                elements_group = [
                                     ui.dropdown(
                                         name=f"{pre}/dataset",
                                         label="Dataset",
                                         required=True,
                                         value=q.client[f"{pre}/dataset"],
                                         choices=[
                                             ui.choice(str(row["id"]), str(row["name"]))
                                             for _, row in df_datasets.iterrows()
                                         ],
                                         trigger=True,
                                         tooltip=tooltip,
                                     )
                                 ] + elements_group

            if len(elements_group) > 0:
                items_list = [
                    ui.separator(
                        name=config_item_name + "_expander", label=make_label(config_item_name, appendix=" settings")
                    )
                ]
            else:
                items_list = []

            items_list += elements_group
        else:
            raise _get_type_annotation_error(config_item_value, type_annotations[config_item_name])

        items += items_list

    q.client[f"{pre}/prev_dataset"] = q.client[f"{pre}/dataset"]

    return items


def parse_ui_elements(
        cfg: Any, q: Q, limit: Union[List, str] = "", pre: str = ""
) -> Any:
    """Sets configuration settings with arguments from app

    Args:
        cfg: configuration
        q: Q
        limit: optional list of keys to limit
        pre: prefix for keys

    Returns:
        Configuration with settings overwritten from arguments
    """

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()
    for k, v in cfg_dict.items():
        if k.startswith("_") or cfg._get_visibility(k) == -1:
            continue

        if (
                len(limit) > 0
                and k not in limit
                and type_annotations[k] in KNOWN_TYPE_ANNOTATIONS
        ):
            continue

        elif type_annotations[k] in KNOWN_TYPE_ANNOTATIONS:
            value = q.client[f"{pre}{k}"]

            if type_annotations[k] == Tuple[str, ...]:
                if isinstance(value, str):
                    value = [value]
                value = tuple(value)
            if type_annotations[k] == str and type(value) == list:
                # fix for combobox outputting custom values as list in wave 0.22
                value = value[0]
            setattr(cfg, k, value)
        elif dataclasses.is_dataclass(v):
            setattr(cfg, k, parse_ui_elements(cfg=v, q=q, limit=limit, pre=pre))
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

    return cfg


def get_dataset_elements(cfg: Any, q: Q) -> List:
    """For a given configuration setting return the according dataset ui components.

    Args:
        cfg: configuration settings
        q: Q

    Returns:
        List of ui elements
    """

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()

    cfg_dict = {key: cfg_dict[key] for key in cfg._get_order()}

    items = []
    for config_item_name, config_item_value in cfg_dict.items():
        # Show some fields only during dataset import
        if config_item_name.startswith("_") or cfg._get_visibility(config_item_name) == -1:
            continue

        if not (
                _check_dependencies(
                    cfg=cfg, pre="dataset/import", k=config_item_name, q=q, dataset_import=True
                )
        ):
            continue
        tooltip = cfg._get_tooltips(config_item_name)

        if type_annotations[config_item_name] in KNOWN_TYPE_ANNOTATIONS:
            if config_item_name in default_cfg.dataset_keys:
                dataset = cfg_dict.copy()
                dataset["path"] = q.client["dataset/import/path"]

                for kk, vv in q.client["dataset/import/cfg"].__dict__.items():
                    dataset[kk] = vv

                for trigger_key in default_cfg.dataset_trigger_keys:
                    if q.client[f"dataset/import/cfg/{trigger_key}"] is not None:
                        dataset[trigger_key] = q.client[
                            f"dataset/import/cfg/{trigger_key}"
                        ]
                if (
                        q.client["dataset/import/cfg/data_format"] is not None
                        and config_item_name == "data_format"
                ):
                    config_item_value = q.client["dataset/import/cfg/data_format"]

                dataset["dataframe"] = q.client["dataset/import/cfg/dataframe"]

                type_annotation = type_annotations[config_item_name]
                poss_values, config_item_value = cfg._get_possible_values(
                    field=config_item_name,
                    value=config_item_value,
                    type_annotation=type_annotation,
                    mode="train",
                    dataset_fn=lambda k, v: (
                        dataset,
                        dataset[k] if k in dataset else v,
                    ),
                )

                if config_item_name == "train_dataframe" and config_item_value != "None":
                    q.client["dataset/import/cfg/dataframe"] = read_dataframe(config_item_value)

                q.client[f"dataset/import/cfg/{config_item_name}"] = config_item_value

                items_list = _get_ui_element(
                    config_item_name,
                    config_item_value,
                    poss_values,
                    type_annotation,
                    tooltip=tooltip,
                    password=False,
                    trigger=(config_item_name in default_cfg.dataset_trigger_keys or config_item_name == "data_format"),
                    q=q,
                    pre="dataset/import/cfg/",
                )
            else:
                items_list = []
        elif dataclasses.is_dataclass(config_item_value):
            items_list = get_dataset_elements(cfg=config_item_value, q=q)
        else:
            raise _get_type_annotation_error(config_item_value, type_annotations[config_item_name])

        items += items_list

    return items


def _get_ui_element(
        config_item_name: str,
        config_item_value: Any,
        poss_values: Any,
        type_annotation: Type,
        tooltip: str,
        password: bool,
        trigger: bool,
        q: Q,
        pre: str = "",
) -> Any:
    """Returns a single ui element for a given config entry

    Args:
        config_item_name: key
        config_item_value: value
        poss_values: possible values
        type_annotation: type annotation
        tooltip: tooltip
        password: flag for whether it is a password
        trigger: flag for triggering the element
        q: Q
        pre: optional prefix for ui key
        get_default: flag for whether to get the default values

    Returns:
        Ui element

    """
    assert type_annotation in KNOWN_TYPE_ANNOTATIONS

    # Overwrite current values with values from yaml
    if pre == "experiment/start/cfg/":
        if q.args["experiment/upload_yaml"] and "experiment/yaml_data" in q.client:
            if (config_item_name in q.client["experiment/yaml_data"].keys()) and (
                    config_item_name != "experiment_name"
            ):
                q.client[pre + config_item_name] = q.client["experiment/yaml_data"][config_item_name]

    if type_annotation in (int, float):
        if not isinstance(poss_values, possible_values.Number):
            raise ValueError(
                "Type annotations `int` and `float` need a `possible_values.Number`!"
            )

        val = q.client[pre + config_item_name] if q.client[pre + config_item_name] is not None else config_item_value

        min_val = (
            type_annotation(poss_values.min) if poss_values.min is not None else None
        )
        max_val = (
            type_annotation(poss_values.max) if poss_values.max is not None else None
        )

        # Overwrite default maximum values with user_settings
        if f"set_max_{config_item_name}" in q.client:
            max_val = q.client[f"set_max_{config_item_name}"]

        if isinstance(poss_values.step, (float, int)):
            step_val = type_annotation(poss_values.step)
        elif poss_values.step == "decad" and val < 1:
            step_val = 10 ** -len(str(int(1 / val)))
        else:
            step_val = 1

        if min_val is None or max_val is None:
            items_list = [
                # TODO: spinbox `trigger` https://github.com/h2oai/wave/pull/598
                ui.spinbox(
                    name=pre + config_item_name,
                    label=make_label(config_item_name),
                    value=val,
                    # TODO: open issue in wave to make spinbox optionally unbounded
                    max=max_val if max_val is not None else 1e12,
                    min=min_val if min_val is not None else -1e12,
                    step=step_val,
                    tooltip=tooltip,
                )
            ]
        else:
            items_list = [
                ui.slider(
                    name=pre + config_item_name,
                    label=make_label(config_item_name),
                    value=val,
                    min=min_val,
                    max=max_val,
                    step=step_val,
                    tooltip=tooltip,
                    trigger=trigger,
                )
            ]
    elif type_annotation == bool:
        val = q.client[pre + config_item_name] if q.client[pre + config_item_name] is not None else config_item_value

        items_list = [
            ui.toggle(
                name=pre + config_item_name,
                label=make_label(config_item_name),
                value=val,
                tooltip=tooltip,
                trigger=trigger,
            )
        ]
    elif type_annotation in (str, Tuple[str, ...]):
        if poss_values is None:
            val = config_item_value

            title_label = make_label(config_item_name)

            items_list = [
                ui.textbox(
                    name=pre + config_item_name,
                    label=title_label,
                    value=val,
                    required=False,
                    password=password,
                    tooltip=tooltip,
                    trigger=trigger,
                    multiline=False,
                )
            ]
        else:
            if isinstance(poss_values, possible_values.String):
                options = poss_values.values
                allow_custom = poss_values.allow_custom
                placeholder = poss_values.placeholder
            else:
                options = poss_values
                allow_custom = False
                placeholder = None

            is_tuple = type_annotation == Tuple[str, ...]

            if is_tuple and allow_custom:
                raise TypeError(
                    "Multi-select (`Tuple[str, ...]` type annotation) and"
                    " `allow_custom=True` is not supported at the same time."
                )

            config_item_value = q.client[pre + config_item_name] if q.client[
                                                                        pre + config_item_name] is not None else config_item_value
            if isinstance(config_item_value, str):
                config_item_value = [config_item_value]

            # `v` might be a tuple of strings here but Wave only accepts lists
            config_item_value = list(config_item_value)

            if allow_custom:
                if not all(isinstance(option, str) for option in options):
                    raise ValueError(
                        "Combobox cannot handle (value, name) pairs for options."
                    )

                items_list = [
                    ui.combobox(
                        name=pre + config_item_name,
                        label=make_label(config_item_name),
                        value=config_item_value[0],
                        choices=list(options),
                        tooltip=tooltip,
                    )
                ]
            else:
                choices = [
                    ui.choice(option, option)
                    if isinstance(option, str)
                    else ui.choice(option[0], option[1])
                    for option in options
                ]

                items_list = [
                    ui.dropdown(
                        name=pre + config_item_name,
                        label=make_label(config_item_name),
                        value=None if is_tuple else config_item_value[0],
                        values=config_item_value if is_tuple else None,
                        required=False,
                        choices=choices,
                        tooltip=tooltip,
                        placeholder=placeholder,
                        trigger=trigger,
                    )
                ]
    else:
        raise AssertionError

    return items_list


def _is_visible(k: str, cfg: Any, q: Q) -> bool:
    """Returns a flag whether a given key should be visible on UI.

    Args:
        k: name of the hyperparameter
        cfg: configuration settings,
        q: Q
    Returns:
        List of ui elements
    """

    visibility = 1

    if cfg._get_visibility(k) > visibility:
        return False

    return True


def _check_dependencies(cfg: Any, pre: str, k: str, q: Q, dataset_import: bool = False):
    """Checks all dependencies for a given key

    Args:
        cfg: configuration settings
        pre: prefix for client keys
        k: key to be checked
        q: Q
        dataset_import: flag whether dependencies are checked in dataset import

    Returns:
        True if dependencies are met
    """

    dependencies = cfg._get_nesting_dependencies(k)

    if dependencies is None:
        dependencies = []
    # Do not respect some nesting during the dataset import
    if dataset_import:
        dependencies = [x for x in dependencies if x.key not in ["validation_strategy"]]
    # Do not respect some nesting during the create experiment
    else:
        dependencies = [x for x in dependencies if x.key not in ["data_format"]]

    if len(dependencies) > 0:
        all_deps = 0
        for d in dependencies:
            if isinstance(q.client[f"{pre}/cfg/{d.key}"], (list, tuple)):
                dependency_values = q.client[f"{pre}/cfg/{d.key}"]
            else:
                dependency_values = [q.client[f"{pre}/cfg/{d.key}"]]

            all_deps += d.check(dependency_values)
        return all_deps > 0

    return True
