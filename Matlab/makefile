#--------------------------------------------------------------
# Makefile for building MATLAB interface
#-------------------------------------------------------------

# Name of MEX source file
MEXNAMES = rb_train_mex rb_test_mex rb_grid_mex

# rb_gen includes and linking
INCLUDE = -I..
CFLAGS_EXTRA ?= -fopenmp
RB_USE_GLIB ?= no
ifeq ($(RB_USE_GLIB),yes)
  INCLUDE += $(shell pkg-config --cflags glib-2.0)
  LIBS += $(shell pkg-config --libs glib-2.0)
  CFLAGS_EXTRA += -DRB_USE_GLIB
endif

#------------------------------------------------------------
# MATLAB configuration

# MATLAB MEX compiler
MEX ?= mex
# Linux
# MEX = /usr/local/matlab/bin/mex
#
# Mac OS X
# MEX = /Applications/MATLAB_R2014a.app/bin/mex
# MEX = /Applications/MATLAB_R2015a.app/bin/mex

# Extra compiler flags 
MEXFLAGS += -O -largeArrayDims
# For debug:
# MEXFLAGS += -g -largeArrayDims

# MATLAB executable app (only for testing)
MATLAB ?= matlab

#------------------------------------------------------------
# Octave configuration

# Octave compiler
MKOCTFILE ?= mkoctfile

# Extra compiler flags 
OCTFLAGS += $(CFLAGS_EXTRA) #-g -Og -Wall -Wextra

# Octave binary (only for testing)
OCTAVE ?= octave --no-gui -H -q

all: matlab

MATLAB_TARGETS := $(foreach mexname,${MEXNAMES},${mexname}_matlab)
matlab: $(MATLAB_TARGETS)
%_matlab: %.cpp
	$(MEX) $(MEXFLAGS) $(LIBS) $(MEXLIBS) $(INCLUDE) -output $* $*.cpp "CFLAGS+=$(CFLAGS_EXTRA)"

octave: $(foreach mexname,${MEXNAMES},${mexname}_octave)
%_octave: %.cpp
	$(MKOCTFILE) --mex $(OCTFLAGS) $(LIBS) $(INCLUDE) $*.cpp

test: test_matlab

test_matlab:
	@trap 'echo MATLAB was killed' 9;\
	rm -f test.log;\
	nohup $(MATLAB) -nodesktop -nodisplay -r "try,testing,catch me,disp(getReport(me));end;exit" -logfile test.log > /dev/null 2>&1; \
	if [ -s test.log ] && grep -q 'Success' test.log; then \
		echo "Test passed!" ;\
	else\
		cat test.log || true ;\
		echo ;\
		echo "Something went wrong or the MATLAB output was not captured.";\
		echo "Consider to send us the file 'test.log'" ;\
		echo "if the software doesn't work as expected." ;\
		exit 1;\
	fi ;\

test_octave:
	@trap 'echo Octave was killed' 9;\
	rm -f test.log;\
	nohup $(OCTAVE) testing.m > test.log 2>&1 ; \
	if [ -s test.log ] && grep -q 'Success' test.log; then \
		echo "Test passed!" ;\
	else\
		cat test.log || true ;\
		echo ;\
		echo "Something went wrong or the Octave output was not captured.";\
		echo "Consider to send us the file 'test.log'" ;\
		echo "if the software doesn't work as expected." ;\
		exit 1;\
	fi ;\

clean:
	rm -f $(foreach mexname,${MEXNAMES},${mexname}.[^c]*) test.log

.PHONY: all matlab octave test test_matlab test_octave clean 
