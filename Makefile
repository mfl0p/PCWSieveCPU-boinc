CC = g++
LD = $(CC)

.SUFFIXES:
.SUFFIXES: .o .c .h .cpp

VER = 23_4_30

APP = PCWSieveCPU-win64-$(VER)

SRC = main.cpp cpu_sieve.cpp cpu_sieve.h sieve_avx512.cpp sieve_64.cpp factor_proth.c factor_proth.h verify_factor.c verify_factor.h putil.c putil.h
OBJ = main.o cpu_sieve.o sieve_avx512.o sieve_64.o factor_proth.o verify_factor.o putil.o

LIBS = libprimesieve.a

BOINC_DIR = C:/mingwbuilds/boinc
BOINC_INC = -I$(BOINC_DIR)/lib -I$(BOINC_DIR)/api -I$(BOINC_DIR) -I$(BOINC_DIR)/win_build
BOINC_LIB = -L$(BOINC_DIR)/lib -L$(BOINC_DIR)/api -L$(BOINC_DIR) -lboinc_api -lboinc

CFLAGS  = -I . -O3 -m64 -Wall -DVERS=\"$(VER)\"
LDFLAGS = $(CFLAGS) -lstdc++ -static -lpthread

all : clean $(APP)

$(APP) : $(OBJ)
	$(LD) $(LDFLAGS) $^ $(LIBS) $(BOINC_LIB) -o $@

main.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ main.cpp

cpu_sieve.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ cpu_sieve.cpp

sieve_avx512.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -mavx512cd -mavx512dq -c -o $@ sieve_avx512.cpp

sieve_64.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ sieve_64.cpp

factor_proth.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ factor_proth.c

verify_factor.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ verify_factor.c

putil.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ putil.c

clean :
	del *.o
	del $(APP).exe

