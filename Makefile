run:
	./main.py
gen: clean
	./melgym/exec/MELGEN ow=o i=./melgym/data/hvac.inp
cor:
	./melgym/exec/MELCOR ./melgym/data/hvac.inp
clean:
	rm -rf ./melgym/out/*