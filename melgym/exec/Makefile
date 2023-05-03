hvac: clean
	./melgen-fusion-186_bdba ../data/hvac.inp
	./melcor-fusion-186_bdba ../data/hvac.inp
sgs: clean
	rm -f ../../melcor_exec.out
	./melgen-fusion-186_bdba ../data/sgs.inp
	./melcor-fusion-186_bdba ../data/sgs.inp
clean:
	rm -f *.DIA
	rm -f *.MES
	rm -f *.OUT
	rm -f *.RST
	rm -f extDIAG
	rm -f MEGDIA
	rm -f MEGOUT
	rm -f MELDIA
	rm -f MELOUT
	rm -f *.dia
	rm -f *.out
	rm -f fort*
cleanall:
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