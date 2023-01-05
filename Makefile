default : Redfield.x

CC_FILES := $(wildcard src/*/*.cc)
OBJ_FILES := $(addprefix obj/,$(notdir $(CC_FILES:.cc=.o)))

CXXLINKER = icpc
CXX = icpc
CXXFLAGS = -g -O2 -DMKL_ILP64 -Wall -std=c++11

Redfield.x : $(OBJ_FILES)
	$(CXXLINKER) $^ -o $@ -L${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

obj/%.o : src/*/%.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $< -I${MKLROOT}/include -qopenmp

clean :
	rm obj/*.o *.x
