EXEC = rb_gen 
SRC = $(wildcard *.c)
OBJS := $(addsuffix .o, $(basename $(SRC)))

CC = gcc
CFLAGS = -g#-fopenmp -Wall

BLAS_INCDIR = 
BLAS_LIBDIR =
BLAS_LIBS = -lblas

LAPACK_INCDIR = 
LAPACK_LIBDIR =
LAPACK_LIBS = -llapack 

INC = -I./ ${LAPACK_INCDIR} ${BLAS_INCDIR} $(shell pkg-config --cflags glib-2.0)

LDFLAGS = ${LAPACK_LIBDIR} ${BLAS_LIBDIR} -lm

LIBS = ${BLAS_LIBS} ${LAPACK_LIBS} $(shell pkg-config --libs glib-2.0)

all: ${EXEC} 

$(EXEC): $(OBJS) Makefile
	$(CC) ${CFLAGS} ${LDFLAGS} $(OBJS) -o $(EXEC) ${LIBS}

%.o : %.c Makefile
	$(CC) ${CFLAGS} ${INC} -c $< -o $@

clean:
	rm -f *.o *.s *.d  

deepclean:
	rm -f *.o *.s *.d *~ ${EXEC}
