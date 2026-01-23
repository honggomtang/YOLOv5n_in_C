CC ?= cc
CFLAGS ?= -O2 -std=c11

INCLUDES = -Icsrc -Iassets -Itests
LIBS = -lm

.PHONY: test_conv0 test_c3 test_c3_debug test_sppf_gen test_sppf test_upsample_gen test_upsample test_layer0_9_gen test_layer0_9 test_layer0_23_gen test_layer0_23 clean

test_conv0:
	$(CC) $(CFLAGS) tests/test_conv0.c csrc/operations/conv2d.c csrc/operations/bn_silu.c $(INCLUDES) $(LIBS) -o tests/test_conv0
	./tests/test_conv0

test_c3:
	$(CC) $(CFLAGS) tests/test_c3.c csrc/blocks/c3.c csrc/operations/conv2d.c csrc/operations/bn_silu.c csrc/operations/bottleneck.c csrc/operations/concat.c $(INCLUDES) $(LIBS) -o tests/test_c3
	./tests/test_c3

test_c3_debug:
	$(CC) $(CFLAGS) tests/test_c3_debug.c csrc/operations/conv2d.c csrc/operations/bn_silu.c csrc/operations/bottleneck.c csrc/operations/concat.c $(INCLUDES) $(LIBS) -o tests/test_c3_debug
	./tests/test_c3_debug

test_sppf_gen:
	python tools/gen_sppf_test_vectors.py --pt assets/yolov5n.pt --out tests/test_vectors_sppf.h --h 8 --w 8

test_sppf:
	$(CC) $(CFLAGS) tests/test_sppf.c csrc/blocks/sppf.c csrc/operations/conv2d.c csrc/operations/bn_silu.c csrc/operations/maxpool2d.c csrc/operations/concat.c $(INCLUDES) $(LIBS) -o tests/test_sppf
	./tests/test_sppf

test_upsample_gen:
	python tools/gen_upsample_test_vectors.py --pt assets/yolov5n.pt --out tests/test_vectors_upsample.h --h 20 --w 20 --c 256

test_upsample:
	$(CC) $(CFLAGS) tests/test_upsample.c csrc/operations/upsample.c $(INCLUDES) $(LIBS) -o tests/test_upsample
	./tests/test_upsample

test_layer0_9_gen:
	python tools/gen_layer0_9_test_vectors.py --pt assets/yolov5n.pt --img /Users/kinghong/Desktop/yolov5/data/images/zidane.jpg --size 64 --out tests/test_vectors_layer0_9.h

test_layer0_9:
	$(CC) $(CFLAGS) tests/test_layer0_9.c csrc/blocks/conv.c csrc/blocks/c3.c csrc/blocks/sppf.c csrc/operations/conv2d.c csrc/operations/bn_silu.c csrc/operations/bottleneck.c csrc/operations/concat.c csrc/operations/maxpool2d.c $(INCLUDES) $(LIBS) -o tests/test_layer0_9
	./tests/test_layer0_9

test_layer0_23_gen:
	python tools/gen_layer0_23_test_vectors.py --pt assets/yolov5n.pt --size 32 --out tests/test_vectors_layer0_23.h

test_layer0_23:
	$(CC) $(CFLAGS) tests/test_layer0_23.c csrc/blocks/conv.c csrc/blocks/c3.c csrc/blocks/sppf.c csrc/operations/conv2d.c csrc/operations/bn_silu.c csrc/operations/bottleneck.c csrc/operations/concat.c csrc/operations/maxpool2d.c csrc/operations/upsample.c $(INCLUDES) $(LIBS) -o tests/test_layer0_23
	./tests/test_layer0_23

clean:
	rm -f tests/test_conv0 tests/test_c3 tests/test_c3_debug tests/test_sppf tests/test_upsample tests/test_layer0_9 tests/test_layer0_23

