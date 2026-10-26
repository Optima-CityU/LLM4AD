# SPDX-License-Identifier: BSD-2-Clause
"""Explicit status of LHNS checkpoint restoration.

The former module was copied from EoH and imported classes that LHNS does not
provide. LHNS does not currently save its current/best function, elite set and
annealing state in a restorable checkpoint. Response replay is a separate feature.
"""


def resume_lhns(lhns, path):
    """Reject incomplete LHNS checkpoints instead of constructing an invalid run."""
    raise NotImplementedError(
        'LHNS does not yet save the state required for checkpoint restoration. '
        'Start a new run; recorded-response replay is a separate capability.'
    )


def resume_eoh(eoh, path):
    """Legacy misnamed entry point; LHNS checkpoint restoration is unsupported."""
    return resume_lhns(eoh, path)
