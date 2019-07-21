INCLUDE_DIR = -I./include
all: libmath_function

libmath_function: ./src/mathFunction.cu
	mkdir -p ./lib
	nvcc -std=c++11 -o lib/libmath_function.so ./src/mathFunction.cu --shared --compiler-options "-fPIC" -arch=sm_61 ${INCLUDE_DIR} -lcublas
