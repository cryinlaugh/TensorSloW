CXX = g++
GCC = gcc
CFLAGS = -O3

#cd .. && make all && cd unitest

build: all

run:
	./testconvrand
all: testconvfp testconvbp testconvrand
testconvbp: ../mnist.o ../cnnConvolutionImp.o ../cnnPoolingImp.o ../blob.o testconvbp.o /opt/OpenBlas/lib/libopenblas.a
		$(CXX) $(CFLAGS) $^ -o $@ -I.  -L/opt/OpenBLAS/lib -lopenblas -lpthread
testconvfp: ../mnist.o ../cnnConvolutionImp.o ../cnnPoolingImp.o ../blob.o testconvfp.o /opt/OpenBlas/lib/libopenblas.a
		$(CXX) $(CFLAGS) $^ -o $@ -I.  -L/opt/OpenBLAS/lib -lopenblas -lpthread
testconvrand: ../mnist.o ../cnnConvolutionImp.o ../cnnPoolingImp.o ../blob.o testconvrand.o /opt/OpenBlas/lib/libopenblas.a
		$(CXX) $(CFLAGS) $^ -o $@ -I.  -L/opt/OpenBLAS/lib -lopenblas -lpthread
%.o: %.cpp
		$(CXX) $(CFLAGS) -c $< -I..
%.o: %.c
		$(GCC) $(CFLAGS) -c $< -I..

clean:
		rm *.o testconvbp testconvfp testconvrand
		rm -rf log/*

