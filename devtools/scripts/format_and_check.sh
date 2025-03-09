#!/usr/bin/env bash
set -e -v
ruff format chempropstereo
ruff check --fix chempropstereo
