#!/bin/bash

for year in {2013..2019}
do
  python3.6 -u prepare_data.py --year $year &
done
wait
