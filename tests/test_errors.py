# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path

import attrs
import h5py
import pytest

from proface.tools.preprocessor import subset
from proface.tools.preprocessor.subset.subset import Config


@pytest.fixture
def h5_file():
    h5 = h5py.File.in_memory()
    h5.create_group("/elements")
    h5["/nodes/numbers"] = [1, 2, 3]
    h5["/nodes/coordinates"] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
    h5["/sets/element/foobar"] = [1, 2, 3]
    h5["/sets/node/baz"] = [1, 2, 3]
    return h5


@pytest.fixture
def h5_pth():
    return Path("foobar")


@pytest.fixture
def valid_job():
    c = Config(subdomain="foobar", internal_interface="interface_internal")
    return attrs.asdict(c)


def test_dummy_run(h5_file, h5_pth, valid_job):
    """this checks that the dummy config does not raises errors"""
    subset.main(job=valid_job, job_path=h5_pth, h5=h5_file)


def test_extra_key_warn(h5_file, h5_pth, valid_job, caplog):
    """this checks that warning for extra keys is logged"""
    job = valid_job
    job["foo"] = "bar"
    job["baz"] = []
    subset.main(job=valid_job, job_path=h5_pth, h5=h5_file)
    assert len(caplog.record_tuples) == 1
    ((logger_name, log_level, message),) = caplog.record_tuples
    assert logger_name == "proface.tools.preprocessor.subset.subset"
    assert log_level == logging.WARNING
    assert message.endswith("Unexpected keys in job configuration: foo, baz.")


def test_main_args_types(h5_file, h5_pth):
    with pytest.raises(TypeError):
        subset.main(job="{}", job_path=h5_pth, h5=h5_file)
    with pytest.raises(TypeError):
        subset.main(job={}, job_path="h5_pth", h5=h5_file)
    with pytest.raises(TypeError):
        subset.main(job={}, job_path=h5_pth, h5="h5_file")


def test_main_job_schema(h5_file, h5_pth, valid_job):
    # missing key
    job = valid_job.copy()
    del job["subdomain"]
    with pytest.raises(subset.SubsetError) as excinfo:
        subset.main(job=job, job_path=h5_pth, h5=h5_file)
    assert str(excinfo.value).endswith(
        "required field missing @ subset.subdomain"
    )

    # invalid subdomain type
    job = valid_job.copy()
    job["subdomain"] = []
    with pytest.raises(subset.SubsetError) as excinfo:
        subset.main(job=job, job_path=h5_pth, h5=h5_file)
    assert str(excinfo.value).endswith(
        "invalid value for type, expected str @ subset.subdomain"
    )

    # invalid internal_interface type
    job = valid_job.copy()
    job["internal_interface"] = []
    with pytest.raises(subset.SubsetError) as excinfo:
        subset.main(job=job, job_path=h5_pth, h5=h5_file)
    assert str(excinfo.value).endswith(
        "invalid value for type, expected str @ subset.internal_interface"
    )


def test_missing_subdomain(h5_file, h5_pth, valid_job):
    job = valid_job.copy()
    job["subdomain"] = "missing"
    with pytest.raises(subset.SubsetError) as excinfo:
        subset.main(job=job, job_path=h5_pth, h5=h5_file)
    assert str(excinfo.value) == "Subdomain element set 'missing' not found."


def test_nset_collision(h5_file, h5_pth, valid_job):
    valid_job["internal_interface"] = "baz"
    with pytest.raises(subset.SubsetError) as excinfo:
        subset.main(job=valid_job, job_path=h5_pth, h5=h5_file)
    assert (
        str(excinfo.value)
        == "Internal interface name (baz) already present in '/sets/node'."
    )
