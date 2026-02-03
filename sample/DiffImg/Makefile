# Makefile for Image Comparison Tool

CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)

# 如果pkg-config找不到opencv，尝试手动指定路径
ifeq ($(OPENCV_FLAGS),)
    OPENCV_INCLUDE = -I/usr/local/include/opencv4 -I/usr/include/opencv4
    OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d -lopencv_highgui
    OPENCV_FLAGS = $(OPENCV_INCLUDE) $(OPENCV_LIBS)
endif

TARGET = image_comparison
SOURCE = image_comparison.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
