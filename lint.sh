#!/bin/sh

flake8 *py wongutils/*/*py
pylint --rcfile=setup.cfg *py wongutils/*/*py

