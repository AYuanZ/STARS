/*
 * Copyright (c) 2023 Georgios Alexopoulos
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cuda_defs.h"
#include "cinfo.h"

#ifndef _STARS_CLIENT_H
#define _STARS_CLIENT_H

extern void continue_with_lock(void);
extern void continue_with_lock_v2(double datasize, double blocks, double threads, int slo);
extern void initialize_client(void);
extern void end_event(void);
extern void create_SM_table(int smcount, int sm_index);

extern double get_current_time_ms(void);
extern int scheduler(double predict_duration_part, int ksid, double submit_time, double con_urgent_level, int smcount, int con_allsmcount, double con_all_throughput, int bs);
extern KS* mps_continue_with_lock(int smcount, int ksid, int slo, double submit_time);
extern KS* mps_continue_with_lock_init(int smcount, int ksid, int slo);

extern void release_lock(void);

#endif /* _STARS_CLIENT_H */
