#!/usr/bin/env bash
pytest tests/liblinodenet/test_performance.py::test_spectral_norm_forward \
  -n 0 --no-cov \
  --benchmark-group-by="func,param:device,param:shape" \
  --benchmark-save=base \
  --benchmark-sort=median \
  --benchmark-compare \
