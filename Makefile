CXX = g++
GCC = gcc
CFLAGS = -O3

all: mnist.o cnnConvolutionImp.o cnnPoolingImp.o blob.o /opt/OpenBlas/lib/libopenblas.a
#		$(CXX) $(CFLAGS) $^ -o $@ -I.  -L/opt/OpenBLAS/lib -lopenblas -lpthread
%.o: %.cpp
		$(CXX) $(CFLAGS) -c $< -I.
%.o: %.c
		$(GCC) $(CFLAGS) -c $< -I.
run:
	make -C ./unitest run

clean:
		rm *.o

