#!/bin/bash

start=`date +%s`

for lr in 1e-04 3e-04 1e-03
do	
	for lambda_r in 1e-02 3e-02 1e-01
	do
		for lambda_o in 1e-03 3e-03 1e-02
		do
			python3.8 -u train.py \
			--year "$1" \
			--cuda "$2" \
			--theta_d 20 \
			--epochs 100 \
			--random_seed 0 \
			--lr $lr \
			--lambda_r $lambda_r \
			--lambda_o $lambda_o
		done
	done
done

end=`date +%s`

echo $((end-start))
