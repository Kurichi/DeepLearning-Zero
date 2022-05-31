CC = g++
CFLAGS = -std=c++2a
CFLAGS += -fopenmp
TARGET = DLZero
INCDIR = -I/usr/include/python3.8
INCDIR += -lpython3.8
INCDIR += -I./Inc

$(TARGET) : main.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(INCDIR)

all: clean $(TARGET)

clean:
	-rm -f $(OBJS) $(TARGET) *.d *.o
