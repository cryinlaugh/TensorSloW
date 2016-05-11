my_sgemm:	my_sgemm.c
		gcc -O3 -o my_sgemm my_sgemm.c /opt/OpenBlas/lib/libopenblas.a
main:	main.c
		gcc -O3 -o main main.c /opt/OpenBlas/lib/libopenblas.a
clean:
		rm *.o main my_sgemm
