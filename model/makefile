CC = clang
CFLAGS = -Wall -Wextra -O3 -std=c11
LDFLAGS = -lm

TARGET = main.out
SRCS = main.c loader.c random.c tensor.c
OBJS = $(SRCS:.c=.o)
HEADERS = loader.h random.h tensor.h

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS)

loader.o: loader.c loader.h tensor.h
tensor.o: tensor.c tensor.h random.h
random.o: random.c random.h
main.o: main.c loader.h tensor.h random.h

.PHONY: all clean
