#!/bin/sh

jupyter nbconvert --to html $1/notebooks/*.ipynb --output-dir=notebooks/
