#!/usr/bin/env bash
pytest-benchmark compare --histogram .benchmarks/histograms/ --group-by="func,param:device,param:shape" --name "trial"
