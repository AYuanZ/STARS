# SPDX-License-Identifier: Apache-2.0

CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=gnu99 -fPIC -D_FORTIFY_SOURCE=2

RELEASED_SOURCES := cli.c comm.c common.c hook.c
RELEASED_HEADERS := comm.h common.h cuda_defs.h

.PHONY: all list check clean

# This staged release intentionally omits internal dependencies required by
# cli.c, comm.c, and hook.c. Build the self-contained utility module only.
all: common.o

common.o: common.c common.h
	$(CC) $(CFLAGS) -c $< -o $@

list:
	@printf '%s\n' $(RELEASED_SOURCES) $(RELEASED_HEADERS)

check: common.o
	@! grep -nP '[\x{4e00}-\x{9fff}]' $(RELEASED_SOURCES) $(RELEASED_HEADERS)

clean:
	$(RM) common.o
