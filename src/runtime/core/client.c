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
 *
 *
 * The stars client. It implements the stars-related logic/threads
 * that we inject into a CUDA application and which manages the interactions
 * with the stars scheduler on behalf of the application.
 */


#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <pthread.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <errno.h>
#include <math.h>

#include "comm.h"
#include "common.h"
#include "client.h"
#include "cuda_defs.h"
#include "lrmodel.h"

void *client_fn(void *arg __attribute__((unused)));
void *release_early_fn(void *arg __attribute__((unused)));
void *release_fn(void *arg __attribute__((unused)));

pthread_t release_thread_tid;
pthread_t client_tid;
pthread_t release_early_thread_tid;
pthread_mutex_t global_mutex;
pthread_cond_t own_lock_cv;
pthread_cond_t release_early_cv;
pthread_cond_t release_cv;
sem_t got_initial_sched_status;
// struct message req_lock_msg = {0};
struct reqmessage req_lock_msg_v2 = {0};
CUcontext cuda_ctx;
int rsock;
int scheduler_on;
int release_early_check_interval = 0.1;
int own_lock;
int need_lock;
int did_work;
uint64_t stars_client_id;
char nvscheduler_socket_path[STARS_SOCK_PATH_MAX];

int prmp_lock = 0;
int start_lock = 0;

// MPS global state
HashTable* Tablesm = NULL;
struct reqmessage req_lock_msg = {0};
char all_ksid_sm_str[256] = {0};
int KS_L = 0;  // Selected kernel-segment window length for the current task
int selected_sm = 0; // Selected SM count
double con_urgent_level = 0.0; // Urgency of colocated tasks
int con_allsmcount = 0; // Total SM count of colocated tasks
double con_all_throughput = 0.0; // Aggregate throughput of colocated tasks
double duration_param = 0.0; // Predicted duration parameter for the current task
int task_slo_all = 0; // SLO of the current task
int allks_num = 0;  // Total number of loaded kernel segments

int g_bs_kernel_counts[MAX_BS_COUNT_INDEX]; // Kernel/profile row count for each batch size
int g_bs_sm[MAX_BS_COUNT_INDEX];

// SM_POOL capacity is defined by the shared protocol header
// Global array of candidate SM counts
int SM_POOL[MAX_SM_POOL_SIZE];
// Number of entries currently stored in SM_POOL
int g_sm_pool_count = 0;

/* Spatial utility weight from Eq. (22); latency receives 1-beta_sched. */
static double beta_sched = 0.5;


// Head of the regression-data linked list
Node* head = NULL;
// Regression weights and bias
double bias, w1, w2;
double beta[4];
int train_data_num = 0;



static void cuda_sync_context(void) {
	CUresult cu_err = CUDA_SUCCESS;

	pending_kernel_window = 1;
	cu_err = real_cuCtxSetCurrent(cuda_ctx);
	cuda_driver_check_error(cu_err, CUDA_SYMBOL_STRING(cuCtxSetCurrent));
	cu_err = real_cuCtxSynchronize();
	cuda_driver_check_error(cu_err, CUDA_SYMBOL_STRING(cuCtxSynchronize));
}


void create_SM_table(int smcount, int sm_index)
{
	// Store this SM count in the global candidate pool
    if (g_sm_pool_count < MAX_SM_POOL_SIZE) {
        SM_POOL[g_sm_pool_count] = smcount;
        g_sm_pool_count++;
    } else {
        log_warn("[client] SM_POOL is full. Cannot add new smcount %d.", smcount);
        return; // Stop if the candidate pool is full
    }

	char *value;
	char filename[256];

	value = getenv(ENV_CLIENT_DIR);

	if (value == NULL) {
		log_warn("[client] ENV_CLIENT_DIR not set");
		return;
	}




	int target_bs[5] = {1, 2, 4, 8, 16};
	if (sm_index < 0 || sm_index >= 5) {
		log_warn("[client] Invalid sm_index: %d", sm_index);
	}else{
		g_bs_sm[target_bs[sm_index]] = smcount;
		log_info("[client] Loading kernel DB for SM count: %d", smcount);
	}




	int num_configs = 5;
	int lastline = 1;
    for (int i = 0; i < num_configs; i++) {
		lastline = 1;
		// Profile path for this SM count and batch size
		snprintf(filename, sizeof(filename), "%s%d/ks_part/ks_summary_sm_%d.csv", value, target_bs[i], smcount);
		// log_info("[client] Loading kernel DB from %s", filename);
        lastline = load_kernel_db(Tablesm, smcount, target_bs[i], filename);
		g_bs_kernel_counts[target_bs[i]] = lastline;
		// log_info("[client] Loaded kernel DB for lastline=%d", lastline);
    }

	// allks_num = lastline;

}

double get_current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}



KS* mps_continue_with_lock_init(int smcount, int ksid, int initslo)
{
	int bs = initslo % 100;
	int slo = initslo / 100;
	// int id_to_find = smcount * 100 + ksid;
	int id_to_find = ksid * 10000 + smcount * 100 + bs;
	// log_info("[client]--------id to find: %d", id_to_find);
	KernelInfo* ks_info = find_ks_info(id_to_find, Tablesm);
	KS* ks = malloc(sizeof(KS));
	ks->kernel_nums = ks_info->kernel_num;
	ks->smcount = smcount;
	return ks;

}


int schedulerp(int ksid, double submit_time, double con_urgentlevel, int smcount, int con_allsmcount, double con_all_throughput, int bs)
{
	// TODO: Return multiple kernel-segment identifiers
	int best_sm = SM_POOL[0];
	double max_efficiency_score = -1.0;

	// TODO: Compute the current task's elapsed time
	double spent_time= get_current_time_ms() - submit_time;
	log_debug("[Client]--290---------Task elapsed: %.2f ms,---current time %.2f---------submission time %.2f", spent_time,  get_current_time_ms(), submit_time);


	// TODO: Compare colocated-task urgency before allowing more interference, then adjust for the current task's urgency.
	log_debug("[Client]--199---------Current ksid---%d----------%d", ksid, g_bs_kernel_counts[bs]);
	double true_progress = ksid * 1.0 / g_bs_kernel_counts[bs];
	double slo_progress = spent_time / task_slo_all;
	log_debug("[Client]--202---------Current ksid---%d----------%d", ksid, task_slo_all);

    // This metric determines whether the aggressive policy is acceptable
	double SFR = slo_progress / true_progress;


	// Standalone durations for different SM allocations
	// TODO: Replace this with a dynamic lookup
	int solo_dur_ns =  find_ks_info((ksid * 10000 + g_bs_sm[bs] *100 + bs), Tablesm)->duration_ns;

	log_debug("[Client]--212---------Current ksid---%d----------%d", ksid, solo_dur_ns);


	int max_duration = find_ks_info((ksid * 10000 + 26 * 100 + bs), Tablesm)->duration_ns;
	log_debug("[Client]--308---------Task standalone duration: %d ns, -----SM%d-------BS%d", solo_dur_ns, g_bs_sm[bs], bs);


	double max_predicted_duration = 0;

	char *env_value1 = getenv(SH);
	int sh_value1 = atoi(env_value1);  // Convert the environment value to an integer

	if (sh_value1 == 1){
		max_predicted_duration = max_duration * (1.3 + 0.1 * 26 / 4.0) * ((con_allsmcount + 26) / 26.0) * 0.2;
	}
	else if(sh_value1 == 2){
		max_predicted_duration = max_duration * (2.5 + 0.5 * 26 / 4.0) * ((con_allsmcount + 26) / 26.0) * 0.2;
	}
	else if (sh_value1 == 3){
		max_predicted_duration = max_duration * (1.3 + 0.1 * 26 / 4.0) * ((con_allsmcount + 26) / 26.0) * 0.2;;
	}else{
		max_predicted_duration = max_duration * (1.5);
		log_warn("[Client]--256---------Workload selector S is unset or invalid; using the default prediction model");
	}

	// log_debug("[Client]--218---------Task standalone duration: %d ns, -----SM%d-------BS%d", solo_dur_ns, g_bs_sm[bs], bs);
	// 1. Predict the duration for each candidate partition
	for(int i=0; i<(g_sm_pool_count-1); i++){

		int sm_candidate = SM_POOL[i];

		int id_to_find = ksid * 10000 + sm_candidate * 100 + bs; // bs = slo/1000
		// log_info("[client]--------id to find: %d", id_to_find);
		KernelInfo* ks_info = find_ks_info(id_to_find, Tablesm);
		int duration_ns = ks_info->duration_ns;
		// Look up the standalone duration, then apply the interference model to obtain T_pred
		// double predicted_duration = duration_ns * 1.15 - sm_candidate * 0.01  - predict_duration_part + con_allsmcount;

		double predicted_duration = 0.0;
		char *env_value = getenv(SH);
		int sh_value = atoi(env_value);  // Convert the environment value to an integer

		if (sh_value == 1){
			predicted_duration = duration_ns * ((1.3 + 0.1 * (sm_candidate / 4.0)) * ((con_allsmcount + sm_candidate) / 26.0) * 0.2);
			log_debug("[Client]-250----Candidate SMs: %d, measured duration=%d, predicted duration=%f", sm_candidate, duration_ns, predicted_duration);
		}
		else if(sh_value == 2){
			predicted_duration = duration_ns * ((2.5 + 0.5 * (sm_candidate / 4.0)) * ((con_allsmcount + sm_candidate) / 26.0) * 0.2);
		}
		else if (sh_value == 3){
			predicted_duration = duration_ns * ((1.3 + 0.1 * (sm_candidate / 4.0)) * ((con_allsmcount + sm_candidate) / 26.0) * 0.2);;
		}else{
			predicted_duration = duration_ns * (1.15);
			// log_warn("[Client]--256---------Workload selector S is unset or invalid; using the default prediction model");
		}

		// Define the penalty term
		double dram = ks_info->dram_demand;
		double l2 = ks_info->l2_demand;
		double urgent_level = SFR * (7 / dram + 3 / l2);
		double v_pressure = (0.7 * dram +  0.3 * l2) * sm_candidate / 2600;
		// log_debug("dram_demand=%f----l2_demand=%f",dram, l2);
		// log_debug("Current penalty=%f----colocated penalty=%f", urgent_level, con_urgentlevel);
		double penalty = 0.45 * con_urgentlevel / urgent_level * v_pressure;

		// Throughput contribution
		double throughput_contribution =  (bs * max_duration) / (predicted_duration);

		// Latency inflation
		double latency_inflation = solo_dur_ns / predicted_duration;

		// Legacy score interpretation: throughput contribution divided by performance loss
		double efficiency_score = beta_sched * throughput_contribution
			+ (1.0 - beta_sched) * latency_inflation - penalty;


		// log_debug("[Client]-250----Candidate SMs: %d, measured duration%d, predicted duration: %f ns, throughput contribution: %f, penalty: %f, efficiency score: %f",
		// 	sm_candidate, duration_ns, predicted_duration, throughput_contribution, penalty, efficiency_score);

		if(efficiency_score > max_efficiency_score){
            max_efficiency_score = efficiency_score;
            best_sm = sm_candidate;
		}

		log_debug("[Client]-250----Candidate SMs: %d, measured duration=%d, predicted duration: %f ns, throughput contribution: %f, penalty: %f, efficiency score: %f, selected SMs: %d",
			sm_candidate, duration_ns, predicted_duration, throughput_contribution, penalty, efficiency_score, best_sm);

		// log_info("[Client]-----Efficiency score with %d SMs: %.4f", sm_candidate, efficiency_score);

	}

	return best_sm;
}



/**
 * Spatial scheduling policy for selecting an SM allocation.
 * @brief Evaluate candidate SM allocations for the current kernel segment.
 * @return Selected SM count.
 */
int scheduler(double predict_duration_part, int ksid, double submit_time, double con_urgentlevel, int smcount, int con_allsmcount, double con_all_throughput, int bs)
{
	// TODO: Return multiple kernel-segment identifiers

	// int sm_options[5] = {6, 8, 14, 20, 28}; // Example set of five candidate SM counts

	int best_sm = SM_POOL[0];
	double max_efficiency_score = -1.0;

	// TODO: Compute the current task's elapsed time
	double spent_time= get_current_time_ms() - submit_time;
	log_debug("[Client]--201---------Task elapsed: %.2f ms,---current time %.2f---------submission time %.2f", spent_time,  get_current_time_ms(), submit_time);

	// TODO: Compare colocated-task urgency before allowing more interference, then adjust for the current task's urgency.
	double true_progress = ksid / g_bs_kernel_counts[bs];
	double slo_progress = spent_time / task_slo_all;
    // This metric determines whether the aggressive policy is acceptable
	double SFR = true_progress / slo_progress;
	double urgrnt_cor = (1 - SFR) - con_urgentlevel;
	log_debug("[Client]--209---------Task urgency: %.4f, colocated urgency: %.4f, urgency difference: %.4f",
		(1 - SFR), con_urgentlevel, urgrnt_cor);
	if (urgrnt_cor < -0.1){
		return smcount;
	}

	int solo_duratoin_ns =  find_ks_info((ksid * 10000 + g_bs_sm[bs] *100 + bs), Tablesm)->duration_ns;

	log_debug("[Client]--218---------Task standalone duration: %d ns, -----SM%d-------BS%d", solo_duratoin_ns, g_bs_sm[bs], bs);
	// 1. Predict the duration for each candidate partition
	for(int i=0; i<(g_sm_pool_count-1); i++){

		int sm_candidate = SM_POOL[i];
		// int id_to_find = sm_candidate * 100 + ksid;
		int id_to_find = ksid * 10000 + sm_candidate * 100 + bs; // bs = slo/1000
		// log_info("[client]--------id to find: %d", id_to_find);
		KernelInfo* ks_info = find_ks_info(id_to_find, Tablesm);
		int duration_ns = ks_info->duration_ns;

		// Look up the standalone duration, then apply the interference model to obtain T_pred
		// double predicted_duration = duration_ns * 1.15 - sm_candidate * 0.01  - predict_duration_part + con_allsmcount;

		double predicted_duration = 0.0;
		char *env_value = getenv(SH);
		int sh_value = atoi(env_value);  // Convert the environment value to an integer

		// if (sh_value == 1){
		// 	predicted_duration = duration_ns * ((1.3 + 0.1 * (sm_candidate / 4)) + ((con_allsmcount + sm_candidate) / 26) * 0.3);
		// }
		// else if(sh_value == 2){
		// 	predicted_duration = duration_ns * ((2.5 + 0.2 * (sm_candidate / 4)) + ((con_allsmcount + sm_candidate) / 26));
		// }
		// else if (sh_value == 3){
		// 	predicted_duration = duration_ns * ((1.3 + 0.1 * (sm_candidate / 4)) + ((con_allsmcount + sm_candidate) / 26) * 0.15);
		// }else{
		// 	predicted_duration = duration_ns * 1.5;
		// log_warn("[Client]--256---------Workload selector S is unset or invalid; using the default prediction model");
		// }

		if (sh_value == 1){
			predicted_duration = duration_ns * ((1.3 + 0.1 * (sm_candidate / 4)) * ((con_allsmcount + sm_candidate) / 26) * 0.2);
		}
		else if(sh_value == 2){
			predicted_duration = duration_ns * ((2.5 + 0.5 * (sm_candidate / 4)) * ((con_allsmcount + sm_candidate) / 26) * 0.2);
		}
		else if (sh_value == 3){
			predicted_duration = duration_ns * ((1.3 + 0.1 * (sm_candidate / 4)) * ((con_allsmcount + sm_candidate) / 26) * 0.2);;
		}else{
			predicted_duration = duration_ns * (1.5);
			log_warn("[Client]--256---------Workload selector S is unset or invalid; using the default prediction model");
		}


		// Penalize interference with the throughput of colocated tasks
		double penalty = con_all_throughput *  predict_duration_part / predicted_duration;

		// Throughput contribution
		// double throughput_contribution =  (sm_candidate * predicted_duration) / (26 * task_slo_all);
		double throughput_contribution =  (bs * 26.0) / (predicted_duration * sm_candidate);


		// Performance loss depends on latency inflation and SLO urgency
		// Urgency
		// double urgency_factor = 1.0 / (task_slo_all - spent_time);
		// double urgency_boost = 1.0 / (1.0 + exp(-(spent_time - task_slo_all * 0.8)/500.0));

		// Latency inflation
		double latency_inflation = solo_duratoin_ns / predicted_duration;

		// Legacy score interpretation: throughput contribution divided by performance loss
		double efficiency_score = 0.5 * throughput_contribution  + 0.5 * latency_inflation;


		log_debug("[Client]-250----Candidate SMs: %d, measured duration%d, predicted duration: %f ns, throughput contribution: %f, penalty: %f, efficiency score: %f",
			sm_candidate, duration_ns, predicted_duration, throughput_contribution, penalty, efficiency_score);

		if(efficiency_score > max_efficiency_score){
            max_efficiency_score = efficiency_score;
            best_sm = sm_candidate;
		}

		// log_info("[Client]-----Efficiency score with %d SMs: %.4f", sm_candidate, efficiency_score);

	}

	// Random SM selection
	// int pool_size = sizeof(SM_POOL) / sizeof(SM_POOL[0]);
	// int random_index = rand() % pool_size; // Uniform index in [0, pool_size - 1]
	// best_sm = SM_POOL[random_index];

	// Greedy SM selection: always choose the largest allocation
	// best_sm =26;


	if (con_allsmcount >= 32){
		best_sm = g_bs_sm[bs];
	}

	best_sm = g_bs_sm[bs];

	return best_sm;
}



/* Scheduling with MPS enabled:
   At each kernel-segment boundary or new request,
   select the context corresponding to the desired SM allocation.
   TODO:--------------
   1. Validate context selection.
   2. Measure interference between concurrent kernel segments and fit an offline model.
   3. Evaluate the scheduling algorithm after the preceding steps are complete.
*/
KS* mps_continue_with_lock(int smcount, int ksid, int initslo, double submit_time)
{
	int slo = initslo / 100;
	int bs = initslo % 100;
	// int id_to_find = smcount * 100 + ksid;
	int id_to_find = ksid * 10000 + smcount * 100 + bs;
	log_info("[client]--------id to find: %d", id_to_find);
	KernelInfo* ks_info = find_ks_info(id_to_find, Tablesm);



	if (ks_info != NULL) {
		// Access members only after validating the pointer
		log_info("Successfully found kernel. ksid=%d, Duration is: %d\n", id_to_find, ks_info->duration_ns);
		// ... Additional operations ...
	} else {
		// Handle a missing profile record
		log_warn("Could not process kernel %d as it was not found in the database.\n", id_to_find);
	}
	// Information returned to the interception layer
	KS* ks = malloc(sizeof(KS));
	ks->kernel_nums = ks_info->kernel_num;
	ks->smcount = smcount;

	true_or_exit(pthread_mutex_lock(&global_mutex) == 0);

	// log_info("[Client]--------own_lock %d", own_lock);

	req_lock_msg.slo = slo;
	req_lock_msg.ks = *ks_info;
	req_lock_msg.smcount = ks->smcount;         // SM count used by this client
	req_lock_msg.submit_time = submit_time;
	req_lock_msg.all_ks_num = g_bs_kernel_counts[bs];


	int solo_duratoin_ns[MAX_SM_POOL_SIZE] = {0};
	// 1. Look up the standalone duration for each candidate partition
	for(int i=0; i<(g_sm_pool_count-1); i++){
		int sm_candidate = SM_POOL[i];
		int id_to_find = ksid * 10000 + sm_candidate * 100 + bs; // bs = slo/1000
		solo_duratoin_ns[i] = find_ks_info(id_to_find, Tablesm)->duration_ns;
	}
	solo_duratoin_ns[g_sm_pool_count]  =find_ks_info((ksid * 10000 + g_bs_sm[bs] *100 + bs), Tablesm)->duration_ns;
	memcpy(req_lock_msg.solo_duratoin_ns, solo_duratoin_ns, sizeof(int) * MAX_SM_POOL_SIZE); // Standalone durations for each SM allocation
	memcpy(req_lock_msg.SM_POOL, SM_POOL, sizeof(int) * MAX_SM_POOL_SIZE); // Available SM candidates
	req_lock_msg.g_sm_pool_count = g_sm_pool_count; // Number of SM candidates
	req_lock_msg.exp_sm = g_bs_sm[bs]; // Expected SM count for this client



	task_slo_all = slo;

	log_info("[Client]----147-----Current task KSID---------%d", req_lock_msg.ks.ks_id);

	/*
		* The application may comprise multiple threads. We must
		* request the lock only once on behalf of the whole app.
		*/



	true_or_exit(write_whole(rsock, &req_lock_msg, sizeof(req_lock_msg)) == sizeof(req_lock_msg));


	log_debug("[Client]----263-----Current task SM count---------%d",  ks->smcount);

	// Wait on own_lock_cv; the client thread broadcasts it when the lock becomes available.
	// Send req_lock_msg. On LOCK_OK, client_fn sets own_lock to 1 and broadcasts own_lock_cv.
	true_or_exit(pthread_cond_wait(&own_lock_cv, &global_mutex) == 0);
	memcpy(ks->outbuffer, all_ksid_sm_str, sizeof(ks->outbuffer));

	log_debug("[Client]----311-----Current task colocated SM count---------%d", con_allsmcount);

	log_info("[Client]----313-----Current task colocated urgency---------%f",  con_urgent_level);


	// TODO: SM scheduling policy
	// int sm_decision = scheduler(duration_paeram, ksid, submit_time, con_urgent_level, smcount, con_allsmcount, con_all_throughput, bs);
	int sm_decision = schedulerp(ksid, submit_time, con_urgent_level, smcount, con_allsmcount, con_all_throughput, bs);

	// int sm_decision = selected_sm;

	// if (getenv(SH) == 1){
	// 	sm_decision = 16;
	// 	KS_L = 20;
	// }

	ks->smcount = sm_decision;
	if ((ksid + KS_L) > g_bs_kernel_counts[bs]) {
		KS_L = g_bs_kernel_counts[bs] - ksid;
	}
	int kernel_nums = 0;
	for (int i = 0; i < KS_L; i++)
	{
		// int id_to_find = sm_decision * 100 + ksid + i;
		int id_to_find = (ksid + i) * 10000 + sm_decision * 100 + bs;
		// log_info("[client]--------id to find: %d", id_to_find);
		KernelInfo* ks_info = find_ks_info(id_to_find, Tablesm);
		kernel_nums += ks_info->kernel_num;
	}

	log_info("[Client]----361-----Selected kernel-segment window length---------%d",  KS_L);
	// Current kernel-segment window length
	ks->ks_len = KS_L;
	log_info("[Client]----364-----Selected kernel-segment window length---------%d",  ks->ks_len);
	// Look up the total kernel count across the selected segments
	ks->kernel_nums = kernel_nums;
	/* We did something. Reset the early release timer. */
	did_work = 1;

	true_or_exit(pthread_mutex_unlock(&global_mutex) == 0);


	return ks;

}


void release_lock(void)
{
	struct reqmessage release_msg = {0};

	release_msg.type = LOCK_RELEASED;
	release_msg.id = stars_client_id;
	release_msg.proc_time = 0.0;
	release_msg.slo = 0.0;

	true_or_exit(pthread_mutex_lock(&global_mutex) == 0);

	log_debug("Releasing the lock due to task finished");
	true_or_exit(write_whole(rsock, &release_msg, sizeof(release_msg)) == sizeof(release_msg));
	log_debug("Sent %s", message_type_string[release_msg.type]);

	true_or_exit(pthread_mutex_unlock(&global_mutex) == 0);
}



/* We use the HOSTNAME environment variable to read the Kubernetes pod name,
 * when we are running on Kubernetes.
 *
 * Only called if we've detected that we're actually running on Kubernetes.
 */
static void read_pod_name(char *pod_name, size_t size)
{
	char *env_hostname = "HOSTNAME";
	char *value;


	value = getenv(env_hostname);
	if (value != NULL) {
		if (strlcpy(pod_name, value, size) >= size)
			log_warn("Pod name is longer than %zu"
				 " characters. Truncating it.", size);
	} else {
		log_debug("Environment variable %s is not set, defaulting to"
			  " \"none\"", env_hostname);
		strlcpy(pod_name, "none", size);
	}
}


/*
 * The namespace of a Pod is accessible by default on Kubernetes through a
 * certain mounted volume:
 *
 * https://stackoverflow.com/a/42449618
 *
 * It's our best shot, so why not take it?
 */
static void read_pod_namespace(char *pod_namespace, size_t size)
{
	char *k8s_pod_ns_file = "/var/run/secrets/kubernetes.io/serviceaccount/namespace";
	FILE *fp;


	fp = fopen(k8s_pod_ns_file, "r");
	if (!fp) {
		log_warn("Couldn't open file %s to read Pod namespace",
			 k8s_pod_ns_file);
		goto out_ns_none;
	} else {
		if (!fgets(pod_namespace, size, fp)) {
			log_warn("Couldn't read the Pod namespace from %s",
				 k8s_pod_ns_file);
			true_or_exit(fclose(fp) == 0);
			goto out_ns_none;
		}
	}

	true_or_exit(fclose(fp) == 0);
	return;

out_ns_none:
	strlcpy(pod_namespace, "none", size);
}


/*
 * Spawn all stars-related threads, bootstrap the client.
 *
 * In more detail:
 * 1. Initialize all locking primitives
 * 2. Create the client thread.
 * 3. Create the early releaser thread.
 * 4. Fill in the globally visible req_lock_msg, that the
 *    app threads will send to the stars-scheduler to request
 *    the GPU lock.
 */
void initialize_client(void)
{
	char *beta_val = getenv("STARS_BETA_SCHED");
	if (beta_val != NULL) {
		char *beta_end = NULL;
		errno = 0;
		double parsed_beta = strtod(beta_val, &beta_end);
		if (errno != 0 || beta_end == beta_val || *beta_end != '\0' ||
			!isfinite(parsed_beta) || parsed_beta < 0.0 || parsed_beta > 1.0) {
			log_fatal("Invalid STARS_BETA_SCHED='%s'; expected a number in [0,1]", beta_val);
		}
		beta_sched = parsed_beta;
	}
	log_info("Spatial scheduling weight: beta_sched=%.3f, latency_weight=%.3f",
		beta_sched, 1.0 - beta_sched);

	scheduler_on = 0;
	cuda_ctx = NULL;
	own_lock = 0;
	need_lock = 0;
	prmp_lock = 0;

	true_or_exit(pthread_cond_init(&release_cv, NULL) == 0); // Signal used to release the lock on task completion
	true_or_exit(pthread_cond_init(&own_lock_cv, NULL) == 0);
	true_or_exit(pthread_cond_init(&release_early_cv, NULL) == 0);
	true_or_exit(pthread_mutex_init(&global_mutex, NULL) == 0);
	true_or_exit(sem_init(&got_initial_sched_status, 0, 0) == 0);

	/* Client thread. */
	true_or_exit(pthread_create(&client_tid, NULL, client_fn, NULL) == 0);

	/* Ensure the client thread has received the initial scheduler status */
	true_or_exit(RETRY_INTR(sem_wait(&got_initial_sched_status)) == 0);

	// true_or_exit(pthread_create(&release_early_thread_tid, NULL,
	// 	     release_early_fn, NULL) == 0);

	true_or_exit(pthread_create(&release_thread_tid, NULL,
		     release_fn, NULL) == 0);


	memset(&req_lock_msg_v2, 0, sizeof(req_lock_msg_v2));
	req_lock_msg_v2.type = REQ_LOCK;
	req_lock_msg_v2.id = stars_client_id;
	req_lock_msg_v2.proc_time = 0.0;
	req_lock_msg_v2.slo = 0.0;



	memset(&req_lock_msg, 0, sizeof(req_lock_msg));
	req_lock_msg.type = REQ_LOCK;
	req_lock_msg.id = stars_client_id;
	req_lock_msg.slo = 0;
	req_lock_msg.smcount = 0;


	Tablesm = create_hash_table(HASH_TABLE_SIZE);

}


/* The stars client main thread.
 *
 * Does the following:
 * 1. Registers client to the stars-scheduler
 * 2. Listens for messages from the stars-scheduler on a persistent connection
 */
void *client_fn(void *arg __attribute__((unused)))
{
	struct reqmessage in_msg;
	struct reqmessage out_msg;

	CUresult cu_err = CUDA_SUCCESS;

	memset(&out_msg, 0, sizeof(out_msg));
	out_msg.id = 1234;
	out_msg.slo = 0.0;
	out_msg.smcount = 0;

	/*
	 * Block every signal for this thread. We want the main thread of the
	 * application to catch all signals.
	 */
	sigset_t signal_set;
	true_or_exit(sigfillset(&signal_set) == 0);
	true_or_exit(pthread_sigmask(SIG_SETMASK, &signal_set, NULL) == 0);

	cu_err = real_cuInit(0);
	cuda_driver_check_error(cu_err, CUDA_SYMBOL_STRING(cuInit));
	if (cu_err != CUDA_SUCCESS)
		log_fatal("cuInit failed when initializing client");

	if (getenv("KUBERNETES_SERVICE_HOST")) {
		read_pod_namespace(out_msg.pod_namespace, sizeof(out_msg.pod_namespace));
		read_pod_name(out_msg.pod_name, sizeof(out_msg.pod_name));
	} else {
		strlcpy(out_msg.pod_namespace, "none", sizeof(out_msg.pod_namespace));
		strlcpy(out_msg.pod_name, "none", sizeof(out_msg.pod_name));
	}

	log_debug("STARS_POD_NAME = %s", out_msg.pod_name);
	log_debug("STARS_POD_NAMESPACE = %s", out_msg.pod_namespace);

	true_or_exit(stars_get_scheduler_path(nvscheduler_socket_path) == 0);

	out_msg.type = REGISTER;

	true_or_exit(stars_connect(&rsock, nvscheduler_socket_path) == 0);
	true_or_exit(write_whole(rsock, &out_msg, sizeof(out_msg)) == sizeof(out_msg));
	log_debug("Sent %s", message_type_string[out_msg.type]);

	/*
	 * Obtain the inital stars-scheduler status
	 */
	true_or_exit(stars_receive_block(rsock, &in_msg, sizeof(in_msg)) == sizeof(in_msg));
	switch (in_msg.type) {
	case SCHED_ON:
		log_debug("Received %s", message_type_string[in_msg.type]);

		true_or_exit(sscanf(in_msg.data, "%" SCNx64, &stars_client_id) == 1);
		log_info("Successfully initialized stars GPU");
		log_info("Client ID = %016" PRIx64, stars_client_id);
		scheduler_on = 1;
		own_lock = 0;
		need_lock = 0;
		break;

	case SCHED_OFF:
		log_debug("Received %s", message_type_string[in_msg.type]);

		true_or_exit(sscanf(in_msg.data, "%" SCNx64, &stars_client_id) == 1);
		log_info("Successfully initialized stars GPU");
		log_info("Client ID = %016" PRIx64, stars_client_id);
		scheduler_on = 0;
		own_lock = 1;
		need_lock = 0;

		break;
	default:
		log_fatal("Got message with type (%d) instead of initial"
			  " stars-scheduler status", (int)in_msg.type);
		break;
	}

	/* The ID will not change henceforth. Fill it in now. */
	memset(&out_msg, 0, sizeof(out_msg));
	out_msg.id = stars_client_id;

	true_or_exit(sem_post(&got_initial_sched_status) == 0);

	while (1) {
		true_or_exit(stars_receive_block(rsock, &in_msg, sizeof(in_msg)) == sizeof(in_msg));
		true_or_exit(pthread_mutex_lock(&global_mutex) == 0);

		switch (in_msg.type) {
		case LOCK_OK:
			log_debug("Received %s", message_type_string[in_msg.type]);

			need_lock = 0;
			own_lock = 1;
			did_work = 1; /* Restart the early release timer to avoid race */
			prmp_lock = 0; // The client is no longer preempted

			// The scheduler sends the identifiers and SM counts of all colocated tasks
			// Fields independent of the scheduling decision
			// memcpy(all_ksid_sm_str, in_msg.signature, sizeof(all_ksid_sm_str));


			// Read the current task's selected window length
			KS_L = in_msg.KS_L;
			selected_sm = in_msg.select_sm;
			// Read the urgency of colocated tasks
			con_urgent_level = in_msg.urgent_level;
			con_allsmcount = in_msg.all_smcount;
			con_all_throughput = in_msg.all_throughput;
			// Read the predicted duration parameter for the current task
			duration_param = in_msg.duration_param;

			// log_info("All current ksid_sm entries---------%s", all_ksid_sm_str);

			true_or_exit(pthread_cond_broadcast(&own_lock_cv) == 0);

			break;
		case LOCK_RELEASED:
			own_lock = 0;
			prmp_lock = 0;
			did_work =0;
			break;
		case DROP_LOCK:
			log_debug("Received %s", message_type_string[in_msg.type]);
			log_debug("Current own_lock == %d", own_lock);
			if (own_lock == 1) { /* Sanity check */
				cuda_sync_context(); /* Ensure all submitted work done */
				own_lock = 0; /* Block work submission */
				prmp_lock = 1; /* The client has been preempted and placed in the waiting queue. */
			}

			break;
		case SCHED_ON:
			log_debug("Received %s", message_type_string[in_msg.type]);

			if (!scheduler_on) { /* WAS OFF, NOW ON */
				log_debug("Scheduler status changed to ON");
				scheduler_on = 1;
				need_lock = 0;
				own_lock = 0;
			} else log_debug("Scheduler status did not change, doing nothing");

			break;
		case SCHED_OFF:
			log_debug("Received %s", message_type_string[in_msg.type]);

			if (scheduler_on) { /* WAS ON, NOW OFF */
				log_debug("Scheduler status changed to OFF");
				scheduler_on = 0;
				own_lock = 1;
				need_lock = 0;
				true_or_exit(pthread_cond_broadcast(&own_lock_cv) == 0);
			}
			break;

		default:
			log_warn("Unknown message type (%d)",
				 (int)in_msg.type);
			break;
		}

		/* Done with this messsage */
		true_or_exit(pthread_mutex_unlock(&global_mutex) == 0);

	}
}



void *release_fn(void *arg __attribute__((unused))){
	struct reqmessage release_msg = {0};

	release_msg.type = LOCK_RELEASED;
	release_msg.id = stars_client_id;
	release_msg.proc_time = 0.0;
	release_msg.slo = 0.0;

	true_or_exit(pthread_mutex_lock(&global_mutex) == 0);
	while (1) {
		did_work = 0;
		int ret2 = pthread_cond_wait(&release_cv, &global_mutex);
		if (ret2 < 0) {
			log_debug("Releasing pthread_cond_wait Error!");
		}else{
			// if (!scheduler_on || !own_lock) continue;

			// if (did_work) {
			// 	did_work = 0;
			// 	continue;
			// }
			// if (start_lock == 0){
			// 	continue;
			// }

			log_debug("Releasing the lock due to task finished");
			true_or_exit(write_whole(rsock, &release_msg, sizeof(release_msg)) == sizeof(release_msg));

			if (did_work)
			{
				did_work = 0;
				continue;
			}

			log_debug("Sent %s", message_type_string[release_msg.type]);
		}

	}
}


// The early-release thread returns the lock when no work remains during the current time quantum.
// STARS does not rely on a time quantum here; release the lock when task completion is detected.
void *release_early_fn(void *arg __attribute__((unused)))
{
	struct reqmessage release_msg = {0};
	struct timespec timer_end_ts = {0, 0};
	struct timespec cuda_sync_start_time = {0, 0};
	struct timespec cuda_sync_complete_time = {0, 0};
	struct timespec cuda_sync_duration = {0, 0};
	int ret;
	unsigned int elapsed_ms;
	nvmlReturn_t nvml_ret;
	nvmlDevice_t nvml_dev;
	nvmlUtilization_t nvml_util;

	release_msg.type = LOCK_RELEASED;
	release_msg.id = stars_client_id;
	release_msg.proc_time = 0.0;
	release_msg.slo = 0.0;

	/*
	 * Block every signal for this thread. We want the main thread of the
	 * application to catch all signals.
	 */
	sigset_t signal_set;
	true_or_exit(sigfillset(&signal_set) == 0);
	true_or_exit(pthread_sigmask(SIG_SETMASK, &signal_set, NULL) == 0);

	if (nvml_ok) {
		nvml_ret = real_nvmlInit();
		if (nvml_ret != NVML_SUCCESS) {
			log_warn("nvmlInit failed with %d", (int)nvml_ret);
			goto check_nvml_ret;
		}
		nvml_ret = real_nvmlDeviceGetHandleByIndex(0, &nvml_dev);
		if (nvml_ret != NVML_SUCCESS) {
			log_warn("nvmlDeviceGetHandleByIndex returned %d",
				 (int)nvml_ret);
		}
check_nvml_ret:
		if (nvml_ret != NVML_SUCCESS) nvml_ok = 0;
	}
	true_or_exit(pthread_mutex_lock(&global_mutex) == 0);

	while (1) {
		did_work = 0;
		true_or_exit(clock_gettime(CLOCK_REALTIME, &timer_end_ts) == 0);
		timer_end_ts.tv_sec += release_early_check_interval;
wait_remainder:
		ret = pthread_cond_timedwait(&release_early_cv, &global_mutex,
			                     &timer_end_ts);
		/* We've locked global_mutex */
		if (ret == ETIMEDOUT) {
			if (!scheduler_on || !own_lock) continue;
			if (did_work) {
				did_work = 0;
				continue;
			}
			if (start_lock == 0){
				continue;
			}
			/*
			 * At this point, the client has not submitted any new
			 * work to the GPU within the last interval.
			 *
			 * However, it could be the case that the client has
			 * already submitted enough work on the GPU to keep it
			 * busy and doesn't need to submit more for now.
			 *
			 * Since "doing work" means keeping the GPU busy, we
			 * use NVML to get the GPU utilization rate and
			 * deduce whether the client is actually idle or not.
			 */
			if (nvml_ok) {
				nvml_ret = real_nvmlDeviceGetUtilizationRates(nvml_dev, &nvml_util);
				if (nvml_ret != NVML_SUCCESS) {
					/*
					 * Don't consider NVML failing a fatal
					 * thing.
					 *
					 * However, this is a bad thing
					 * to happen, because if we've
					 * reached this point then we've found
					 * the NVML symbols, which indicates a
					 * deeper error.
					 */
					log_warn("nvmlInit failed with %d", (int)nvml_ret);
					nvml_ok = 0; /* Stop using NVML */
					continue;
				} else {
					log_debug("GPU Utilization = %u %%", nvml_util.gpu);
					if (nvml_util.gpu > 0) {
						log_debug("Early release timer elapsed but we are not idle");
						continue;
					}
				}
			} else { /* FALLBACK method in case NVML fails */
				/*
				 * The idea is the following:
				 *
				 * Check if cuCtxSynchronize() takes more than
				 * 100 ms.
				 *
				 * This is an empirical threshold above
				 * which we consider that we've been doing work
				 * on the GPU for the past interval instead of
				 * being idle.
				 */
				true_or_exit(clock_gettime(CLOCK_MONOTONIC,
					&cuda_sync_start_time) == 0);
				cuda_sync_context();
				true_or_exit(clock_gettime(CLOCK_MONOTONIC,
					&cuda_sync_complete_time) == 0);
				timespecsub(&cuda_sync_complete_time, &cuda_sync_start_time, &cuda_sync_duration);

				elapsed_ms = ((unsigned int)cuda_sync_duration.tv_sec * 1000)
					     + (cuda_sync_duration.tv_nsec / 1000000);
				if (elapsed_ms >= 100) {
					log_debug("Early release timer elapsed but we are not idle");
					continue;
				}
			}

			/* IDLE */
			log_debug("Releasing the lock early due to inactivity");
			true_or_exit(write_whole(rsock, &release_msg, sizeof(release_msg)) == sizeof(release_msg));
			own_lock = 0;
			free_hash_table(Tablesm);
			log_debug("Sent %s", message_type_string[release_msg.type]);
		} else if (ret != 0) { /* BAD */
			errno = ret;
			log_fatal_errno("pthread_cond_timedwait() failed");
		} else { /* Condition variable was signaled */
			if (did_work) continue;
			else goto wait_remainder; /* Spurious wakeup */
		}
	}
}
