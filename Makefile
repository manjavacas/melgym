all: clean drl

rand:
	python3 examples/rand_controller.py

drl:
	python3 examples/drl_controller.py

clean:
	rm -rf best_models *log*.txt logs *.csv melgym/out