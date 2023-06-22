all: clean run
run:
	./main.py
gen: clean
	./melgym/exec/MELGEN ow=o i=./melgym/data/base.inp
cor:
	./melgym/exec/MELCOR ./melgym/data/base.inp
clean:
	rm -rf ./melgym/out/*
	rm -rf ./metrics/