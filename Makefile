all: clean drl

rand:
	python3 examples/rand_controller.py

drl:
	python3 examples/drl_controller.py

clean:
	rm -rf best_model *log*.txt *log.csv* logs monitor.csv melgym/out