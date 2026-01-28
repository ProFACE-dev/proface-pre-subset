# SPDX-FileCopyrightText: 2026 ProFACE developers
#
# SPDX-License-Identifier: MIT

from importlib.metadata import entry_points

from proface.tools.preprocessor.subset import main


def test_entrypoint() -> None:
    # check that preprocessor entrypoint is correctly defined and loadable
    (ep,) = entry_points(name="subset", group="proface.preprocessor.tools")
    main_ep = ep.load()
    assert main_ep is main
