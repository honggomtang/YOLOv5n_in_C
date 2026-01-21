CC ?= cc
CFLAGS ?= -O2 -std=c11

INCLUDES = -Icinclude -Iassets -Itests
LIBS = -lm

.PHONY: test_conv0 clean

test_conv0:
	$(CC) $(CFLAGS) csrc/test_conv0.c csrc/conv2d_ref.c csrc/bn_silu_ref.c $(INCLUDES) $(LIBS) -o tests/test_conv0
	./tests/test_conv0

clean:
	rm -f tests/test_conv0

