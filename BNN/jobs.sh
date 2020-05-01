#!/bin/bash


for seed in {2..40..2}
	do
		python code.py --dataset=boston --alpha=-inf --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=boston --alpha=0 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=boston --alpha=0.5 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=boston --alpha=1 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=boston --alpha=inf --epochs=500 --K=100 --seed=$seed

	done

for seed in {2..40..2}
	do
		python code.py --dataset=concrete --alpha=-inf --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=concrete --alpha=0 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=concrete --alpha=0.5 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=concrete --alpha=1 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=concrete --alpha=inf --epochs=500 --K=100 --seed=$seed

	done

for seed in {2..40..2}
	do
		python code.py --dataset=wine --alpha=-inf --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=wine --alpha=0 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=wine --alpha=0.5 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=wine --alpha=1 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=wine --alpha=inf --epochs=500 --K=100 --seed=$seed

	done

for seed in {2..40..2}
	do
		python code.py --dataset=yacht --alpha=-inf --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=yacht --alpha=0 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=yacht --alpha=0.5 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=yacht --alpha=1 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=yacht --alpha=inf --epochs=500 --K=100 --seed=$seed

	done

for seed in {2..40..2}
	do
		python code.py --dataset=QSAR --alpha=-inf --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=QSAR  --alpha=0 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=QSAR  --alpha=0.5 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=QSAR  --alpha=1 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=QSAR  --alpha=inf --epochs=500 --K=100 --seed=$seed

	done

for seed in {2..40..2}
	do
		python code.py --dataset=power --alpha=-inf --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=power --alpha=0 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=power --alpha=0.5 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=power --alpha=1 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=power --alpha=inf --epochs=500 --K=100 --seed=$seed

	done

for seed in {2..40..2}
	do
		python code.py --dataset=forest --alpha=-inf --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=forest  --alpha=0 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=forest  --alpha=0.5 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=forest  --alpha=1 --epochs=500 --K=100 --seed=$seed
		python code.py --dataset=forest  --alpha=inf --epochs=500 --K=100 --seed=$seed

	done


for seed in {2..10..2}
	do
		python code.py --dataset=protein --alpha=-inf --epochs=100 --K=10 --seed=$seed
		python code.py --dataset=protein --alpha=0 --epochs=100 --K=10 --seed=$seed
		python code.py --dataset=protein --alpha=0.5 --epochs=100 --K=10 --seed=$seed
		python code.py --dataset=protein --alpha=1 --epochs=100 --K=10 --seed=$seed
		python code.py --dataset=protein --alpha=inf --epochs=100 --K=10 --seed=$seed

	done

python code.py --dataset=year --alpha=-inf --epochs=40 --K=10 --seed=2
python code.py --dataset=year --alpha=0 --epochs=40 --K=10 --seed=2
python code.py --dataset=year --alpha=0.5 --epochs=40 --K=10 --seed=2
python code.py --dataset=year --alpha=1 --epochs=40 --K=10 --seed=2
python code.py --dataset=year --alpha=inf --epochs=40 --K=10 --seed=2



for alpha in -100 -10 -5 5 10 100 1000 10000 100000 1000000 10000000
	do
		for seed in {2..40..2}
			do
				python code.py --dataset=boston --alpha=$alpha --epochs=500 --K=100 --seed=$seed
				python code.py --dataset=yacht --alpha=$alpha --epochs=500 --K=100 --seed=$seed
			done
	done
