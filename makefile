# Makefile for posix and gcc

# Note on old compilers  *cough*  DoC  *cough* you might need -std=c++0x instead
CPPFLAGS = -std=c++11 -stdlib=libstdc++ -framework Opencl -I include

#LDFLAGS = -L /opt/local/lib

GLOBALFILES = src/heat.cpp

# Turn on optimisations
CPPFLAGS += -O2


make_world : src/make_world.cpp
	clang++ $(CPPFLAGS) $^ $(GLOBALFILES) -o $@

step_world : src/step_world.cpp
	clang++ $(CPPFLAGS) $^ $(GLOBALFILES) -o $@

render_world : src/render_world.cpp
	clang++ $(CPPFLAGS) $^ $(GLOBALFILES) -o $@

test_opencl : src/test_opencl.cpp
	clang++ $(CPPFLAGS) $^ $(GLOBALFILES) -o $@
	
step_world_v1_lambda : src/tp709/step_world_v1_lambda.cpp
	clang++ $(CPPFLAGS) $^ $(GLOBALFILES) -o $@
	
step_world_v2_function : src/tp709/step_world_v2_function.cpp
	clang++ $(CPPFLAGS) $^ $(GLOBALFILES) -o $@
	
step_world_v3_opencl : src/tp709/step_world_v3_opencl.cpp
	clang++ $(CPPFLAGS) $^ $(GLOBALFILES) -o $@

all : make_world step_world render_world test_opencl step_world_v1_lambda step_world_v2_function step_world_v3_opencl
	./make_world 100 0.1 | ./step_world 0.1 1000 > step.txt
	./make_world 100 0.1 | ./step_world_v1_lambda 0.1 1000 > step_v1.txt
	./make_world 100 0.1 | ./step_world_v2_function 0.1 1000 > step_v2.txt
	./make_world 100 0.1 | ./step_world_v3_opencl 0.1 1000 > step_v3.txt
	diff step.txt step_v1.txt
	diff step.txt step_v2.txt
	diff step.txt step_v3.txt
