# SPDX-FileCopyrightText: 2026 Lorenzo Rusnati
# SPDX-FileCopyrightText: 2026 Stefano Miccoli
#
# SPDX-License-Identifier: MIT

import array
import logging
from pathlib import Path
from typing import Any, Never

import attrs
import cattrs
import h5py
import numpy as np
import numpy.typing as npt
from proface.preprocessor import PreprocessorError

logger = logging.getLogger(__name__)


class SubsetError(PreprocessorError):
    """plugin error conditions"""


@attrs.define(frozen=True)
class Config:
    subdomain: str
    internal_interface: str = "INTERNAL_INTERFACE"


def _structure(job: dict[str, str]) -> Config:
    """convert dict to Cstructured Config class"""

    c = cattrs.Converter(forbid_extra_keys=False)

    @c.register_structure_hook
    def validate(val: Any, _) -> str:  # noqa: ANN001, ANN401
        if not isinstance(val, str):
            msg = f"{val!r} not a string"
            raise ValueError(msg)  # noqa: TRY004
        return val

    try:
        config = c.structure(job, Config)
    except cattrs.ClassValidationError as exc:
        msg = "Error parsing job configuration: "
        msg += "; ".join(cattrs.transform_error(exc, path="subset"))
        raise SubsetError(msg) from exc
    logger.debug("Subset job: %s", config)

    expected = [i.name for i in attrs.fields(Config)]
    unexpected = [i for i in job if i not in expected]
    if unexpected:
        logger.warning(
            "Unexpected keys in job configuration: %s", ", ".join(unexpected)
        )

    return config


def modify(*, job: dict[str, str], job_path: Path, h5: h5py.File) -> None:
    """modify in place h5, computing subset defined in job"""

    # explicit type checking,
    if (
        not isinstance(job, dict)
        or not isinstance(job_path, Path)
        or not isinstance(h5, h5py.File)
    ):
        msg = "Invalid arg types"
        raise TypeError()

    # parse config
    config = _structure(job)

    # create subdomain element set
    if config.subdomain not in h5["/sets/element"]:
        msg = f"Subdomain element set '{config.subdomain}' not found."
        raise SubsetError(msg)
    set_element = np.asarray(h5["/sets/element"][config.subdomain])
    logger.info(
        "Subdomain set %s, %d elements", config.subdomain, len(set_element)
    )

    # check collision of internal interface node set name
    if config.internal_interface in h5["/sets/node"]:
        msg = (
            f"Internal interface name ({config.internal_interface}) "
            "already present in '/sets/node'."
        )
        raise SubsetError(msg)

    # create array.array for subset of nodes of kept/discarded elements
    gnode_type = h5["/nodes/numbers"].dtype
    assert gnode_type.isnative
    arr_gnode_keep = array.array(gnode_type.char)
    arr_gnode_discard = array.array(gnode_type.char)

    # process /elements
    for eltype, group in h5["/elements"].items():
        _process_elements(
            eltype=eltype,
            group=group,
            results=h5["/results"],
            subset=set_element,
            arr_node_keep=arr_gnode_keep,
            arr_node_discard=arr_gnode_discard,
        )

    # process /nodes
    set_node_keep, set_node_discard = _process_nodes(
        group=h5["/nodes"],
        arr_keep=arr_gnode_keep,
        arr_discard=arr_gnode_discard,
    )

    # process /sets
    _process_sets(h5["/sets/element"], set_element)
    _process_sets(h5["/sets/node"], set_node_keep)

    # add internal interface set
    set_interface = np.intersect1d(
        set_node_keep, set_node_discard, assume_unique=True
    )
    h5["sets/node"][config.internal_interface] = set_interface
    logger.info(
        "Internal interface (%s): %d nodes",
        config.internal_interface,
        len(set_interface),
    )


def _process_elements(
    *,
    eltype: str,
    group: h5py.Group,
    results: h5py.Group,
    subset: npt.NDArray,
    arr_node_keep: array.array,
    arr_node_discard: array.array,
) -> None:
    """process /elements"""
    logger.info("Processing %s", group.name)

    numbers = group["numbers"]
    nodes = group["nodes"]

    # compute intersection
    # set_numbers, idx_numbers, _ = np.intersect1d(
    #     numbers, subset, assume_unique=True, return_indices=True
    # )
    msk_numbers = np.isin(numbers, subset, assume_unique=True)
    set_numbers = numbers[msk_numbers]

    # shortcuts
    if len(set_numbers) == 0:
        # No elements in intersection: delete all
        logger.info("- No elements in subset")
        del group.file[group.name]
        # ...and also results
        _subset_results(eltype, results, [], [])
        # ...and update list of dicarded nodes
        arr_node_discard.extend(np.asarray(nodes))
        return
    elif len(set_numbers) == len(numbers):
        # All elements in subset, leave elements as is
        assert np.all(msk_numbers)
        logger.info("- All elements in subset")
        # ... and update list of kept nodes
        arr_node_keep.extend(np.asarray(nodes))
        return

    # FIXME: check group attributes for consistency

    # do subset numbers
    logger.info("- Elements in subset %d / %d", len(set_numbers), len(numbers))
    _replace_data(numbers, set_numbers)

    # do subset incidences
    incidences = group["incidences"]
    set_incidences = np.asarray(incidences)[msk_numbers]
    _replace_data(incidences, set_incidences)

    # do subset nodes
    set_nodes = np.unique(set_incidences, sorted=True)
    logger.info("- Nodes in subset %d / %d", len(set_nodes), len(nodes))
    _replace_data(nodes, set_nodes)
    arr_node_keep.extend(set_nodes)

    # do nodes of discarded elements
    set_nodes_discard = np.unique(
        np.asarray(incidences)[~msk_numbers], sorted=True
    )
    arr_node_discard.extend(set_nodes_discard)

    # do subset results
    _, idx_nodes, _ = np.intersect1d(
        np.asarray(nodes), set_nodes, assume_unique=True, return_indices=True
    )
    assert len(idx_nodes) == len(set_nodes)
    _subset_results(eltype, results, msk_numbers, idx_nodes)


def _process_nodes(
    *,
    group: h5py.Group,
    arr_keep: array.array,
    arr_discard: array.array,
) -> tuple[npt.NDArray, npt.NDArray]:
    """process /nodes group"""
    logger.info("Processing %s", group.name)

    ds_nodes = group["numbers"]
    set_keep = np.unique(arr_keep, sorted=True)
    set_discard = np.unique(arr_discard, sorted=True)
    _, idx_keep, _ = np.intersect1d(
        np.asarray(ds_nodes),
        set_keep,
        assume_unique=True,
        return_indices=True,
    )

    logger.info("- Nodes in subset %d / %d", len(set_keep), len(ds_nodes))
    _replace_data(group["numbers"], set_keep)

    ds_coord = group["coordinates"]
    set_coord = np.asarray(ds_coord)[idx_keep]
    _replace_data(ds_coord, set_coord)

    return set_keep, set_discard


def _process_sets(group: h5py.Group, set_numbers: npt.NDArray) -> None:
    logger.info("Processing %s", group.name)

    for ds in group.values():
        set_ds = np.intersect1d(np.asarray(ds), set_numbers, assume_unique=True)
        _replace_data(ds, set_ds)


def _replace_data(ds: h5py.Dataset, data: npt.NDArray) -> None:
    """replace dataset with new data, delete if data empty"""
    logger.debug(
        "- %s (%d \N{RIGHTWARDS ARROW} %d)", ds.name, len(ds), len(data)
    )

    name = ds.name
    # unlink old dataset
    del ds.file[name]
    # return if no new data
    if len(data) == 0:
        return
    # set provided data
    assert ds.shape[1:] == data.shape[1:]
    assert not ds.attrs, list(ds.attrs)
    ds.file.create_dataset(
        name=name,
        data=data.astype(dtype=ds.dtype, casting="same_value", copy=False),
    )


def _subset_results(
    eltype: str,
    results: h5py.Group,
    idx_numbers: npt.NDArray | list[Never],
    idx_nodes: npt.NDArray | list[Never],
) -> None:
    """subset result values, according do nodal elementa indices"""

    for load_case in results.values():
        for quantity in load_case.values():
            ipgroup = quantity.get("integration_point")
            if ipgroup is not None:
                val = ipgroup[eltype]
                set_val = np.asarray(val)[idx_numbers]
                _replace_data(val, set_val)
            ndgroup = quantity.get("nodal_averaged")
            if ndgroup is not None:
                val = ndgroup[eltype]
                set_val = np.asarray(val)[idx_nodes]
                _replace_data(val, set_val)
