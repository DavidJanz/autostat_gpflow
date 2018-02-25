#!/usr/bin/env bash

mkdir data
curl ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_mlo.txt >> data/co2_raw.txt