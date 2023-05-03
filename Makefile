run: clean
	./main.py

clean:
	rm -f *.DIA
	rm -f *.MES
	rm -f *.OUT
	rm -f *.RST
	rm -f *.DAT
	rm -f *.PTF
	rm -f extDIAG
	rm -f MEGDIA
	rm -f MEGOUT
	rm -f MELDIA
	rm -f MELOUT
	rm -f *.dia
	rm -f *.out
	rm -f fort*
	rm -f melgym/exec/*.DIA
	rm -f melgym/exec/*.MES
	rm -f melgym/exec/*.OUT
	rm -f melgym/exec/*.RST
	rm -f melgym/exec/*.DAT
	rm -f melgym/exec/*.PTF
	rm -f melgym/exec/extDIAG
	rm -f melgym/exec/MEGDIA
	rm -f melgym/exec/MEGOUT
	rm -f melgym/exec/MELDIA
	rm -f melgym/exec/MELOUT
	rm -f melgym/exec/*.dia
	rm -f melgym/exec/*.out
	rm -f melgym/exec/fort*
