run:
	./main.py
gen: clean
	./melgym/exec/MELGEN ow=o i=./melgym/data/simple.inp
cor:
	./melgym/exec/MELCOR ./melgym/data/simple.inp
clean:
	rm -rf ./melgym/out/*