all: cleanmetrics run
rand:
	./rand_controller.py
run:
	./run_experiment.py -conf cfg.json
gen: clean
	./melgym/exec/MELGEN ow=o i=./melgym/data/branch_2.inp
cor:
	./melgym/exec/MELCOR ./melgym/data/branch_2.inp
clean:
	rm -f MEGDIA
	rm -f MELDIA
	rm -f MELMES
	rm -f MEGOUT
	rm -f MELOUT
	rm -f MELPTF
	rm -f MELRST
	rm -f *.DAT
	rm -f extDIAG
cleanout:
	rm -rf ./melgym/out/*
cleantb:
	rm -rf ./tensorboard/*
cleanmetrics:
	rm -rf ./ep_metrics/*
cleanmodels:
	rm -rf ./best_models/*
cleanall: clean cleanout cleantb cleanmetrics