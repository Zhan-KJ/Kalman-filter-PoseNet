#COMPLILER USED
CC = g++
AR = ar cru

#SO FLAG
 SHARE = -shared -fPIC

 TARGET = pycall

#EXE PGM  AND LIB
 all: ${TARGET}

#MAKE RULE
 ${TARGET}:
	$(CC) $@.cpp $(SHARE) $^ -o $@.so

 clean:
	rm -f ${TARGET}.so
