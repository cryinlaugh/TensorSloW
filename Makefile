CXX = g++
GCC = gcc
CFLAGS = -O3
OBJS = mnist.o cnnConvolutionImp.o cnnPoolingImp.o innerprodImp.o blob.o

all: $(OBJS)

%.o: %.cpp
		$(CXX) $(CFLAGS) -c $< -I.
%.o: %.c
		$(GCC) $(CFLAGS) -c $< -I.
run: all
	make -C ./unitest run

clean:
		rm *.o
		make -C ./unitest clean

