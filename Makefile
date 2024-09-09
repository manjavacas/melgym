BRANCH ?= pid

all: cleanmetrics run
gencor: gen cor
rand:
	./rand_controller.py
run:
	./run_drl.py -conf config.yaml
gen: clean
	./melgym/exec/MELGEN ow=o i=./melgym/data/$(BRANCH).inp
cor:
	./melgym/exec/MELCOR ./melgym/data/$(BRANCH).inp
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