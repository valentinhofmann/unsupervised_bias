#!/bin/bash

start=`date +%s`

for lr in 1e-04 3e-04 1e-03
do	
	python3.8 -u train.py \
	--year "$1" \
	--cuda "$2" \
	--theta_d 768 \
	--epochs 100 \
	--lambda_o 0 \
	--lambda_r 0 \
	--random_seed 0 \
	--lr $lr
done

end=`date +%s`

echo $((end-start))
