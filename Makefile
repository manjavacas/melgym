all: clean run
rand:
	./rand_controller.py
run:
	./run_experiment.py -conf cfg.json
gen: clean
	./melgym/exec/MELGEN ow=o i=./melgym/data/simple.inp
cor:
	./melgym/exec/MELCOR ./melgym/data/simple.inp
clean:
	rm -rf ./melgym/out/*
	rm -rf ./metrics/
	rm -f MEGDIA
	rm -f MELDIA
	rm -f MELMES
	rm -f MEGOUT
	rm -f MELOUT
	rm -f MELPTF
	rm -f MELRST
	rm -f *.DAT
	rm -f extDIAG