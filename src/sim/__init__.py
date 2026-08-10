"""The simulation and drafting layer.

Parallel to `src/models/` and `src/eda/` because these are neither models nor analyses:
everything here is numpy over the posterior artifacts `make posteriors` writes, and only
that target and `make stan-game-length` need a CmdStan toolchain.
"""
