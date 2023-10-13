#!/usr/bin/env bash
pytest tests/liblinodenet/test_performance.py -n 0 --no-cov \
  --benchmark-group-by="func,param:device,param:shape" \
  --benchmark-save=base \
  --benchmark-sort=mean \
  -k test_singular_triplet_forward
