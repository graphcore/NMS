CC = g++
CXXFLAGS_LIB = -std=c++14 -g -fPIC -shared -I..
LDLIBS = -lpoplar -lpopops -lpoputil -lpopnn
INC = -I../../
BUILD_DIR = build
SOURCES = nms_custom_op.cpp
TARGET_LIB = $(BUILD_DIR)/nms_custom_op.so

all: create_build_dir nms_custom_op

.PHONY: create_build_dir
create_build_dir: 
	mkdir -p $(BUILD_DIR)

nms_custom_op: nms_custom_op.cpp ../../nms.cpp ../../ipu_utils.cpp
	$(CXX) $? $(INC)  $(LDLIBS) $(CXXFLAGS_LIB) $(ONNX_NAMESPACE)  -Wl,-soname,$@ -o $(TARGET_LIB)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)

