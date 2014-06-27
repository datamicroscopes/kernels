-include config.mk

DEBUG ?= 0

O := out
TOP := $(shell echo $${PWD-`pwd`})

# set the CXXFLAGS
CXXFLAGS := -fPIC -g -MD -Wall -std=c++0x -I$(TOP)/include
CXXFLAGS += -I$(TOP)/../common/include
CXXFLAGS += -I$(TOP)/../mixturemodel/include
ifneq ($(strip $(DEBUG)),1)
	CXXFLAGS += -O3 -DNDEBUG
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

UNAME_S := $(shell uname -s)
TARGETS :=
LIBPATH_VARNAME :=
ifeq ($(UNAME_S),Linux)
	TARGETS := $(O)/libmicroscopes_kernels.so
	LIBPATH_VARNAME := LD_LIBRARY_PATH
endif
ifeq ($(UNAME_S),Darwin)
	TARGETS := $(O)/libmicroscopes_kernels.dylib
	LIBPATH_VARNAME := DYLD_LIBRARY_PATH
endif

all: $(TARGETS)

$(O)/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(O)/libmicroscopes_kernels.so: $(OBJFILES)
	gcc -shared -o $(O)/libmicroscopes_kernels.so $(OBJFILES) $(LDFLAGS)

$(O)/libmicroscopes_kernels.dylib: $(OBJFILES)
	g++ -dynamiclib -o $(O)/libmicroscopes_kernels.dylib $(OBJFILES) $(LDFLAGS)

DEPFILES := $(wildcard out/kernels/*.d)
ifneq ($(DEPFILES),)
-include $(DEPFILES)
endif

.PHONY: clean
clean: 
	rm -rf out
	find microscopes \( -name '*.cpp' -or -name '*.so' -or -name '*.pyc' \) -type f -print0 | xargs -0 rm --

.PHONY: test
test:
	$(LIBPATH_VARNAME)=$$$(LIBPATH_VARNAME):../common/out:../mixturemodel/out:./out PYTHONPATH=$$PYTHONPATH:../common:../mixturemodel:. nosetests

.PHONY: fast_test
fast_test:
	$(LIBPATH_VARNAME)=$$$(LIBPATH_VARNAME):../common/out:../mixturemodel/out:./out PYTHONPATH=$$PYTHONPATH:../common:../mixturemodel:. nosetests -a '!slow'
