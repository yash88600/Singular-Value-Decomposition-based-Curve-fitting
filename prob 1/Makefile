#make file - this is a comment section
 
CC=g++  #compiler
TARGET=main #target file name
INPUT_FILE="main.cpp"
OUTPUT_FILE='main'

all: inverse

run:
	./main

inverse: 
	$(CC) main.cpp -o main -O1 -larmadillo

.PHONY: clean
clean:
	$(RM) ${or ${OF}, main, read_matrix_and_solve}