# Project: Project1
# Makefile created by Dev-C++ 4.9.9.2

CPP  = g++.exe
CC   = gcc.exe
WINDRES = windres.exe
RES  = 
OBJ  = main.o $(RES)
LINKOBJ  = main.o $(RES)
LIBS =  -L"P:/DevCpp/Dev-Cpp/lib" -L"E:/PortableSoftware/Boost1.48/libs"  
INCS =  -I"P:/DevCpp/Dev-Cpp/include" 
CXXINCS =  -I"P:/DevCpp/Dev-Cpp/lib/gcc/mingw32/3.4.2/include"  -I"P:/DevCpp/Dev-Cpp/include/c++/3.4.2/backward"  -I"P:/DevCpp/Dev-Cpp/include/c++/3.4.2/mingw32"  -I"P:/DevCpp/Dev-Cpp/include/c++/3.4.2"  -I"P:/DevCpp/Dev-Cpp/include"  -I"E:/PortableSoftware/Boost1.48" 
BIN  = SharedIDTest.exe
CXXFLAGS = $(CXXINCS)  
CFLAGS = $(INCS)  
RM = rm -f

.PHONY: all all-before all-after clean clean-custom

all: all-before SharedIDTest.exe all-after


clean: clean-custom
	${RM} $(OBJ) $(BIN)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -o "SharedIDTest.exe" $(LIBS)

main.o: main.cp