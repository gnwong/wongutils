#!/bin/sh

flake8 *py wongutils/*/*py tests/*py
pylint --rcfile=setup.cfg *py wongutils/*/*py

