nvcc_options = -ccbin icpc -std=c++17

program : thrust_sort.cu thrust_all.cuh
	nvcc $(nvcc_options) $< -o $@

run : program
	./program

clean :
	rm -f ./program

.PHONY : run clean
