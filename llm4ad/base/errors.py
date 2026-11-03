# SPDX-License-Identifier: BSD-2-Clause
"""Errors that must terminate a search instead of triggering a candidate retry."""


class FatalSearchError(RuntimeError):
    """A failed run contract, such as incompatible replay or checkpoint data."""


class ReplayError(FatalSearchError):
    """A terminal replay failure; search loops must propagate it to the caller."""


class ReplayMismatchError(ReplayError, ValueError):
    """The next request differs from the recorded request."""


class ReplayExhaustedError(ReplayError, EOFError):
    """The search requested more responses than the journal contains."""


class ReplayBackendError(ReplayError):
    """The next journal entry records a backend failure."""

