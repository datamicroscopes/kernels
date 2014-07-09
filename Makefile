-include config.mk

DEBUG ?= 0
NOSE_TIMEOUT ?= 600

O := out
TOP := $(shell echo $${PWD-`pwd`})

CPUCOUNT := $(shell python -c 'import multiprocessing as m; print m.cpu_count()')

# set the CXXFLAGS
CXXFLAGS := -fPIC -g -MD -Wall -std=c++0x -I$(TOP)/include
CXXFLAGS += -I$(TOP)/../common/include
CXXFLAGS += -I$(TOP)/../mixturemodel/include
ifneq ($(strip $(DEBUG)),1)
	CXXFLAGS += -O3 -DNDEBUG
else
	CXXFLAGS += -DDEBUG_MODE
endif
ifneq ($(strip $(DISTRIBUTIONS_INC)),)
	CXXFLAGS += -I$(DISTRIBUTIONS_INC)
endif

# set the LDFLAGS
LDFLAGS := -lprotobuf -ldistributions_shared -lmicroscopes_common -lmicroscopes_mixturemodel
LDFLAGS += -L$(TOP)/../common/out -Wl,-rpath,$(TOP)/../common/out
LDFLAGS += -L$(TOP)/../mixturemodel/out -Wl,-rpath,$(TOP)/../mixturemodel/out
ifneq ($(strip $(DISTRIBUTIONS_LIB)),)
	LDFLAGS += -L$(DISTRIBUTIONS_LIB) -Wl,-rpath,$(DISTRIBUTIONS_LIB) 
endif

SRCFILES := $(wildcard src/kernels/*.cpp) 
OBJFILES := $(patsubst src/%.cpp, $(O)/%.o, $(SRCFILES))

TESTPROG_SRCFILES := $(wildcard test/cxx/*.cpp)
TESTPROG_BINFILES := $(patsubst %.cpp, %.prog, $(TESTPROG_SRCFILES))

TESTPROG_LDFLAGS := $(LDFLAGS)
TESTPROG_LDFLAGS += -L$(TOP)/out -Wl,-rpath,$(TOP)/out
TESTPROG_LDFLAGS += -lmicroscopes_kernels

UNAME_S := $(shell uname -s)
TARGETS :=
LIBPATH_VARNAME :=
ifeq ($(UNAME_S),Linux)
	TARGETS := $(O)/libmicroscopes_kernels.so
	LIBPATH_VARNAME := LD_LIBRARY_PATH
	EXTNAME := so
	SHARED_FLAG := -shared
endif
ifeq ($(UNAME_S),Darwin)
	TARGETS := $(O)/libmicroscopes_kernels.dylib
	LIBPATH_VARNAME := DYLD_LIBRARY_PATH
	EXTNAME := dylib
	SHARED_FLAG := -dynamiclib
endif

all: $(TARGETS)

$(O)/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(O)/libmicroscopes_kernels.$(EXTNAME): $(OBJFILES)
	$(CXX) $(SHARED_FLAG) -o $@ $(OBJFILES) $(LDFLAGS)

%.prog: %.cpp $(O)/libmicroscopes_kernels.$(EXTNAME)
	$(CXX) $(CXXFLAGS) $< -o $@ $(TESTPROG_LDFLAGS)

DEPFILES := $(wildcard out/kernels/*.d)
ifneq ($(DEPFILES),)
-include $(DEPFILES)
endif

.PHONY: clean
clean: 
	rm -rf out test/cxx/*.{d,dSYM,prog}
	find microscopes \( -name '*.cpp' -or -name '*.so' -or -name '*.pyc' \) -type f -print0 | xargs -0 rm -f --

.PHONY: test
test: $(O)/libmicroscopes_kernels.$(EXTNAME)
	python setup.py build_ext --inplace
	$(LIBPATH_VARNAME)=$$$(LIBPATH_VARNAME):../common/out:../mixturemodel/out:./out PYTHONPATH=$$PYTHONPATH:../common:../mixturemodel:. nosetests 

.PHONY: fast_test
fast_test: $(O)/libmicroscopes_kernels.$(EXTNAME)
	python setup.py build_ext --inplace
	$(LIBPATH_VARNAME)=$$$(LIBPATH_VARNAME):../common/out:../mixturemodel/out:./out PYTHONPATH=$$PYTHONPATH:../common:../mixturemodel:. nosetests -a '!slow' 
