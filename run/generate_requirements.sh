#!/usr/bin/env bash
poetry export --without-hashes --output requirements.txt --extras all
poetry export  --without-hashes --output requirements-dev.txt --with dev --extras all
