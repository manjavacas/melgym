all: cleanmetrics run
rand:
	./rand_controller.py
run:
	./run_experiment.py -conf cfg.json
gen: clean
	./melgym/exec/MELGEN ow=o i=./melgym/data/branch_0_v2.inp
cor:
	./melgym/exec/MELCOR ./melgym/data/branch_0_v2.inp
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
cleanmodels:
	rm -rf ./best_models/*
cleanmetrics:
	rm -rf ./ep_metrics/*
cleanall: clean cleanout cleantb cleanmodels cleanmetrics