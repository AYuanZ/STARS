# SPDX-License-Identifier: Apache-2.0
CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -std=gnu99 -fPIC -D_FORTIFY_SOURCE=2
CPPFLAGS += -Isrc/runtime/core -Isrc/profiling -Isrc/modeling -Isrc/scheduler
BUILD := build
CORE := src/runtime/core
SCHEDULER ?= stars
SCHEDULERS := stars FIFO EDF EDF-noP SJF SRTF
ifeq ($(filter $(SCHEDULER),$(SCHEDULERS)),)
$(error Unknown SCHEDULER '$(SCHEDULER)'; choose $(SCHEDULERS))
endif
SCHED_SOURCE := src/scheduler/$(if $(filter stars,$(SCHEDULER)),scheduler.c,scheduler-$(SCHEDULER).c)
COMMON := $(BUILD)/$(CORE)/common.o $(BUILD)/$(CORE)/comm.o
LIB_SRC := $(CORE)/hook.c $(CORE)/client.c src/profiling/cinfo.c src/modeling/lrmodel.c
LIB_OBJ := $(addprefix $(BUILD)/,$(LIB_SRC:.c=.o))
HARDEN := -Wl,-z,defs -Wl,-z,relro -Wl,-z,now

.PHONY: all clean check check-workloads schedulers FORCE
all: $(BUILD)/libstars.so $(BUILD)/stars-scheduler $(BUILD)/starsctl

$(BUILD)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/libstars.so: $(LIB_OBJ) $(COMMON) $(CORE)/libstars-symbols.ld
	$(CC) $(LDFLAGS) $(HARDEN) -shared -Wl,-soname=libstars.so -Wl,--version-script=$(CORE)/libstars-symbols.ld $(filter %.o,$^) -o $@ -ldl -lpthread -lm

# Re-select the binary even when switching back to a cached algorithm.
$(BUILD)/stars-scheduler: $(BUILD)/scheduler-$(SCHEDULER) FORCE
	cp $< $@

$(BUILD)/scheduler-$(SCHEDULER): $(BUILD)/$(SCHED_SOURCE:.c=.o) $(COMMON) $(BUILD)/src/scheduler/queuing.o $(BUILD)/src/modeling/interference.o
	$(CC) $(LDFLAGS) $(HARDEN) $^ -o $@ -lpthread -lm

$(BUILD)/starsctl: $(BUILD)/$(CORE)/cli.o $(BUILD)/$(CORE)/xopt.o $(COMMON)
	$(CC) $(LDFLAGS) $(HARDEN) $^ -o $@

schedulers:
	@set -e; for algorithm in $(SCHEDULERS); do $(MAKE) SCHEDULER=$$algorithm $(BUILD)/scheduler-$$algorithm; done

check: all
	python3 tests/check_host.py
	python3 tests/check_workloads.py

check-workloads:
	python3 tests/check_workloads.py

clean:
	$(RM) -r $(BUILD)

-include $(shell test ! -d $(BUILD) || find $(BUILD) -name '*.d')
