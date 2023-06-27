all: clean run
rand:
	./rand_controller.py
run:
	./run_experiment.py -conf cfg.json
gen: clean
	./melgym/exec/MELGEN ow=o i=./melgym/data/branch.inp
cor:
	./melgym/exec/MELCOR ./melgym/data/branch.inp
clean:
	rm -rf ./melgym/out/*
	rm -rf ./metrics/