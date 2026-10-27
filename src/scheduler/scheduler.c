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
 * The stars scheduler.
 */

#include <dirent.h>
#include <pthread.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/epoll.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <math.h>

#include "comm.h"
#include "common.h"
#include "utlist.h"
#include "queuing.h"
#include "interference.h"

#define STARS_DEFAULT_TQ 200000

#define MAX_QUEUE_SIZE 20 // Maximum number of tasks in the candidate queue

int lock_held;
int must_reset_timer;
pthread_cond_t timer_cv;
pthread_cond_t insert_cv;
pthread_cond_t pop_cv;
pthread_cond_t preemt_cv;
int scheduler_on;
int tq;
unsigned int scheduling_round = 0;

struct reqmessage out_msg = {0};

char nvscheduler_socket_path[STARS_SOCK_PATH_MAX];

pthread_mutex_t global_mutex;
pthread_mutex_t insert_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t pop_mutex = PTHREAD_MUTEX_INITIALIZER;
/* File descriptor for epoll */
int epoll_fd;

// MPS state
// Predefined kernel-segment window lengths
/*
 * Global policy weight used by the temporal risk controller.  It can be
 * overridden for sensitivity studies with STARS_W_SFR.  W_prr is always
 * (1 - W_sfr), so the controller has only one independent weight.
 */
static double W_sfr = 0.7;
/*
 * Half-width of the piecewise-linear risk transition around one.  The
 * manuscript defines the same [1-delta, 1+delta] interval for SFR and PRR.
 * Keep it runtime-configurable so a sensitivity run changes the policy, not
 * the source tree or the workload.
 */
static double risk_delta = 0.2;
const int L_base = 1;
int L_aggressive = 20;



/* Holds the requests for the GPU lock, which we serve in an FCFS manner */
struct stars_request {
	struct stars_client *client;
	struct stars_request *next;
};

/* Scheduling and measurement with MPS enabled:
  The active-task list holds tasks belonging to currently active clients.
  TODO:
  1. Collect interference measurements for offline modeling.
  When a request arrives, record the active clients' kernel-segment features as interference inputs. The interception layer measures the duration at synchronization.
  Alternatively, use the interval between consecutive requests as the segment duration.
*/
Actask active_task_list[MAX_QUEUE_SIZE];
int active_task_count = 0;
// static void insert_actask(Actask in_actask, char *ksid_sm_str);
pthread_mutex_t search_mutex = PTHREAD_MUTEX_INITIALIZER;


// Initialize the candidate queue and its size
Task waiting_task_queue[MAX_QUEUE_SIZE];
int task_queue_size = 0;
Task *current_task = NULL;   // Currently running task
Task *in_task = NULL;       // Task entering the waiting queue

struct stars_client *clients = NULL;
struct stars_request *requests = NULL;  // List of lock requests

void *timer_thr_fn(void *arg __attribute__((unused)));

// void *DP_insert_task(void *arg __attribute__((unused)));

static void bcast_status(void);
static int send_message(struct stars_client *client, struct reqmessage *msg_p);
static int receive_message(struct stars_client *client, struct reqmessage *msg_p);
// static void try_schedule(void);
static int register_client(struct stars_client *client, const struct reqmessage *in_msg);
static int has_registered(struct stars_client *client);
static void client_id_as_string(char *buf, size_t buflen, uint64_t id);
static void delete_client(struct stars_client *client);
static void insert_req(struct stars_client *client);
static void remove_req(struct stars_client *client);


// MPS prediction, scheduling, and measurement helpers
static double predict_duration(Actask now_task, Actask con_tasks);
static double insert_actask(Actask in_actask, double *ur_level, int *allsumcount, double *all_th, char *ksid_sm_str);


// static void try_schedule_v2(void);

static int preempt_scheduling(Task current_task, Task new_task);
static int try_preemt_schedule(Task new_task);
static double get_current_time_ms();

static int has_registered(struct stars_client *client)
{
	return (client->id != STARS_UNREGISTERED_ID);
}

/* Print an stars client ID as a hex string */
static void client_id_as_string(char *buf, size_t buflen, uint64_t id)
{
	if (id == STARS_UNREGISTERED_ID)
		strlcpy(buf, "<UNREGISTERED>", buflen);
	else snprintf(buf, buflen, "%016" PRIx64, id);
}

static void delete_client(struct stars_client *client)
{
	int cfd = client->fd;
	char id_str[HEX_STR_LEN(client->id)];
	struct stars_client *tmp, *c;


	client_id_as_string(id_str, sizeof(id_str), client->id);
	log_info("Removing client %s", id_str);
	remove_req(client);

	/* Remove from clients list */
	LL_FOREACH_SAFE(clients, c, tmp) {
		if (c->fd == client->fd) {
			LL_DELETE(clients, c);
			free(c);
		}
	}

	true_or_exit(epoll_ctl(epoll_fd, EPOLL_CTL_DEL, cfd, NULL) == 0);
	/* See man close(2) for EINTR behavior on Linux */
	if (close(cfd) < 0 && errno != EINTR)
		log_fatal_errno("Failed to close FD %d", cfd);
}



/**
 * Predict the current task's duration using features of active, colocated tasks.
 * @brief Evaluate the empirical linear interference model.
 * @param now_task Features of the current task.
 * @param con_tasks Features of the colocated workload.
 * @return Empirical interference estimate in the original calibration's units.
 */
static double predict_duration(Actask now_task, Actask con_tasks) {
	// Evaluate the predictor using current and colocated workload features
	// The released snapshot uses the original empirical linear coefficients
	double duration = stars_predict_interference(now_task.ks.dram_demand,
        now_task.ks.l2_demand, con_tasks.smcount,
        con_tasks.ks.dram_demand, con_tasks.ks.l2_demand);

	return duration;
}


/**
 * Predict execution duration at a different SM count.
 */
static double predict_sm_duration(double sm_known, int sm_new, double duration) {
	// Evaluate the predictor using current and colocated workload features
	double new_duration = stars_predict_sm_duration(sm_known, sm_new, duration);

	return new_duration;
}

// Piecewise-linear shaping function
double shape_value(double value, double danger_threshold, double safe_threshold) {
    if (value < danger_threshold) return 0.0;
    if (value > safe_threshold) return 1.0;
    return (value - danger_threshold) / (safe_threshold - danger_threshold);
}


/**
 * Temporal scheduling policy for selecting a kernel-segment window.
 * @brief Combine SLO progress and interference progress to choose KS_L.
 * @return Number of kernel segments in the selected execution window.
 */
static int scheduler(Actask *now_task, int allks_num, double submit_time, int task_slo, int is_alone) {
	L_aggressive = allks_num / 5;
	if (L_aggressive < L_base) L_aggressive = L_base;
	int ksid = (now_task->ks.ks_id) / 10000;
	double true_progress = ksid * 1.0 / allks_num;
	double expected_progress = (get_current_time_ms() - submit_time) * 1000000.0 / (now_task->ks.solo_duration_ns);
	double slo_progress = (get_current_time_ms() - submit_time) / (task_slo);

	// SLO progress ratio: actual progress divided by elapsed SLO fraction
	// Compare completed work with the fraction of the latency budget already consumed
    // This metric determines whether the aggressive policy is acceptable
	double SFR = true_progress / slo_progress;

	// Aggregate task urgency and return it to the client for spatial allocation decisions.
	// A task behind its SLO progress has higher scheduling pressure; combine this with colocated-task urgency when selecting SMs.
	now_task->urgent_level = SFR * (7 / now_task->ks.dram_demand + 3 / now_task->ks.l2_demand);

	// Progress ratio: actual progress divided by expected standalone progress
    // Estimate the severity of interference experienced by the task
    // If expected progress substantially exceeds actual progress, interference is high
	double PRR = true_progress / expected_progress;

	int KS_L = L_aggressive;
	double risk_danger = 1.0 - risk_delta;
	double risk_safe = 1.0 + risk_delta;
	if (SFR < risk_danger) KS_L = L_aggressive; // Use the aggressive segment window when the progress ratio is low
	if (is_alone) return  KS_L = L_aggressive; // Use the aggressive segment window when running alone

	// 1. Compute the health score
    double F_sfr = shape_value(SFR, risk_danger, risk_safe);
    double G_prr = shape_value(PRR, risk_danger, risk_safe);
    double health_score = W_sfr * (1 - F_sfr) + (1 - W_sfr) * (1 - G_prr);

	// 2. Compute the continuous window length L*
    double L_continuous = L_base + health_score * (L_aggressive - L_base);

    // 3. Round and clamp the selected window length
    int L_star = (int)round(L_continuous);
	if (L_star < L_base) L_star = L_base;
	if (L_star > L_aggressive) L_star = L_aggressive;

	log_debug("[Scheduler][RiskControl] W_sfr=%.3f W_prr=%.3f SFR=%.6f PRR=%.6f L_star=%d",
		W_sfr, 1.0 - W_sfr, SFR, PRR, L_star);

	/* Return the risk-adaptive value computed above. */
	return L_star;
}



/**
 * Spatial scheduling policy for selecting an SM allocation.
 * @brief Evaluate candidate SM allocations for the current kernel segment.
 * @return Selected SM count.
 */
int schedulerp(Actask *now_task, double predict_duration_part, double con_urgentlevel, int smcount,
	int con_allsmcount, double con_all_throughput, int bs, const int SM_POOL[], int g_sm_pool_count, int solo_duration_ns[], int exp_sm, int all_ksnum)
{
	int best_sm = SM_POOL[0];
	double max_efficiency_score = -1.0;
	double spent_time= get_current_time_ms() - now_task->submit_time;
	double true_progress = (now_task->ks.ks_id) / 10000 / all_ksnum;
	double slo_progress = spent_time / now_task->slo;

	// Standalone durations for different SM allocations
	int solo_dur_ns =  solo_duration_ns[g_sm_pool_count];

	double max_predicted_duration = predict_sm_duration(exp_sm, SM_POOL[g_sm_pool_count -1], now_task->ks.duration_ns);

	// 1. Predict the duration for each candidate partition
	for(int i=0; i<(g_sm_pool_count-1); i++){

		int sm_candidate = SM_POOL[i]; // Candidate SM count
		int duration_ns = solo_duration_ns[i]; // Standalone duration for this SM allocation

		// Look up the standalone duration, then apply the interference model to obtain T_pred
		// double predicted_duration = duration_ns * 1.15 - sm_candidate * 0.01  - predict_duration_part + con_allsmcount;

		double predicted_duration = 0.0;
		char *env_value = getenv(SH);
		int sh_value = atoi(env_value);  // Convert the environment value to an integer


		// Define the penalty term
		double v_pressure = (0.6 * now_task->ks.dram_demand +  0.4 * now_task->ks.l2_demand) * smcount / 2600.0;
		double penalty = con_urgentlevel / now_task->urgent_level * v_pressure;

		// Throughput contribution
		double throughput_contribution =  (bs * 26.0 * max_predicted_duration) / (predicted_duration * sm_candidate);

		// Latency inflation
		double latency_inflation = solo_dur_ns / predicted_duration;

		// Legacy score interpretation: throughput contribution divided by performance loss
		double efficiency_score = 0.5 * throughput_contribution  + 0.5 * latency_inflation - penalty;

		// 	sm_candidate, duration_ns, predicted_duration, throughput_contribution, penalty, efficiency_score);
		if(efficiency_score > max_efficiency_score){
            max_efficiency_score = efficiency_score;
            best_sm = sm_candidate;
		}
	}


	return best_sm;
}



/* MPS: active-task list management
*/
// Current timestamp in milliseconds
static double get_current_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}



// Insert a task into the candidate queue
static double insert_actask(Actask in_actask, double *ur_level, int *allsmcount, double *all_th, char *ksid_sm_str)
{
	// Assume the result buffer has sufficient space
	// A 256-byte buffer holds roughly 60 four-character task entries plus separators

	// Temporary buffer for formatting one task entry
	char task_buffer[10];
	Actask con_tasks = {0};

	true_or_exit(pthread_mutex_lock(&insert_mutex) == 0);
	int ksid_sm = 0;
	int found = 0;
	double urgent = 0.0;
	int allsm = 0;
	double throughput_contribution = 0.01;


	if (active_task_count==0){
		FIFO(in_actask, active_task_list, &active_task_count);
		ksid_sm = 0;
	}
	else{
		for (int i = 0; i < active_task_count; ++i) {
			if (active_task_list[i].client != NULL && active_task_list[i].client->id == in_actask.client->id) {
				// Replace the entry for an existing client
				active_task_list[i].ks = in_actask.ks;
				active_task_list[i].smcount = in_actask.smcount;
				found = 1;
				continue;
			}
			else{
				con_tasks.smcount += active_task_list[i].smcount;
				con_tasks.ks.dram_demand += active_task_list[i].ks.dram_demand;
				con_tasks.ks.l2_demand += active_task_list[i].ks.l2_demand;
				urgent += active_task_list[i].urgent_level;
			}
		}
	}

	if (!found)
	{
		// Append an entry if no matching client was found
		FIFO(in_actask, active_task_list, &active_task_count);
	}


	*ur_level = urgent;
	*allsmcount = con_tasks.smcount;
	*all_th = throughput_contribution;

	double predict_duration_part = predict_duration(in_actask, con_tasks);

	true_or_exit(pthread_mutex_unlock(&insert_mutex) == 0);

	return predict_duration_part;
}





// Insert a task into the candidate queue
static void insert_actask_old(Actask in_actask, char *ksid_sm_str)
{
	// Assume the result buffer has sufficient space
	// A 256-byte buffer holds roughly 60 four-character task entries plus separators
	char result_string[256] = {0}; // Initialize an empty string

	// Temporary buffer for formatting one task entry
	char task_buffer[10];

	true_or_exit(pthread_mutex_lock(&insert_mutex) == 0);
	int ksid_sm = 0;
	int found = 0;
	if (active_task_count==0){
		FIFO(in_actask, active_task_list, &active_task_count);
		ksid_sm = 0;
	}
	else{
		for (int i = 0; i < active_task_count; ++i) {
			if (active_task_list[i].client != NULL && active_task_list[i].client->id == in_actask.client->id) {
				// Replace the entry for an existing client
				active_task_list[i].ks = in_actask.ks;
				active_task_list[i].smcount = in_actask.smcount;
				found = 1;
				continue;
			}

		}
	}

	if (!found)
	{
		// Append an entry if no matching client was found
		log_debug("No existing client found for ID %lu. Adding as a new task.\n", in_actask.client->id);
		FIFO(in_actask, active_task_list, &active_task_count);
	}
	true_or_exit(pthread_mutex_unlock(&insert_mutex) == 0);

}


// Remove tasks belonging to a disconnected client
static void delete_actask(struct stars_client *client)
{
    for (int i = 0; i < active_task_count; i++)
	{
		if (client->fd == active_task_list[i].client->fd)
		{
			 // Shift subsequent entries forward
            for (int j = i; j < active_task_count - 1; j++)
			{
                active_task_list[j] = active_task_list[j + 1];
            }
			active_task_count--;
		}
    }
}

// Remove tasks belonging to a disconnected client
static int release_actask(struct stars_client *client)
{
    for (int i = 0; i < active_task_count; ++i) {
        if (client->fd == active_task_list[i].client->fd) {
            for (int j = i; j < active_task_count - 1; ++j)
                active_task_list[j] = active_task_list[j + 1];
            --active_task_count;
            return 1;
        }
    }
    return 0;
}


/**
 * @brief Write active_task_list to a CSV file.
 * @param filename Output file path.
 * @return Number of records written, or -1 on failure.
 */
int save_active_list_csv(const char* filename) {
    FILE *fp = fopen(filename, "w"); // "w" -> Write Text
    if (!fp) {
        perror("Failed to open file for text writing");
        return -1;
    }

    // Write the CSV header
    fprintf(fp, "client_id,ks_id,smcount\n");

    int records_written = 0;
    for (int i = 0; i < active_task_count; ++i) {
        if (active_task_list[i].client != NULL) {
            // Format and write each record with fprintf
            int result = fprintf(fp, "%llu,%d,%d\n",
                    (unsigned long long)active_task_list[i].client->id,
                    active_task_list[i].ks.ks_id,
                    active_task_list[i].smcount
				);

            if (result < 0) {
                fprintf(stderr, "Error writing record %d to CSV file.\n", i);
                fclose(fp);
                return -1;
            }
            records_written++;
        }
    }

    fclose(fp);
    printf("Successfully wrote %d records to %s (CSV).\n", records_written, filename);
    return records_written;
}




// Remove the client's model request from the request list
static void remove_req(struct stars_client *client)
{
	struct stars_request *tmp, *r;
	if (requests != NULL) {
		/*
		 * This client was holding the GPU lock, as it was the head of
		 * the requests list.
		 */
		if (requests->client->fd == client->fd) lock_held = 0;
	}
	LL_FOREACH_SAFE(requests, r, tmp) {
		if (r->client->fd == client->fd) {
			LL_DELETE(requests, r);
			free(r);
		}
	}
}



static int register_client(struct stars_client *client, const struct reqmessage *in_msg)
{
	int ret;
	struct stars_client *c;
	uint64_t stars_client_id;

	if (has_registered(client)) {
		log_warn("Client %016" PRIx64 " is already registered",
			 client->id);
		return -1;
	}

again:
	stars_client_id = stars_generate_id();
	if (stars_client_id == STARS_UNREGISTERED_ID) /* Tough luck */
		goto again;
	LL_FOREACH(clients, c) {
		if (c->id == stars_client_id) { /* ID clash */
			goto again;
		}
	}

	/*
	 * Store the rest of the client information.
	 */
	client->id = stars_client_id;
	strlcpy(client->pod_name, in_msg->pod_name,
		sizeof(client->pod_name));
	strlcpy(client->pod_namespace, in_msg->pod_namespace,
		sizeof(client->pod_namespace));

	/*
	 * Inform the client of the current status of our current status, as
	 * well as the ID we generated for it.
	 *
	 * It will henceforth present this ID to interact with us.
	 */
	true_or_exit(snprintf(out_msg.data, 16+1, "%016" PRIx64, stars_client_id) == 16);
	out_msg.type = scheduler_on ? SCHED_ON : SCHED_OFF;

	if ((ret = send_message(client, &out_msg)) < 0)
		goto out_with_msg;

out_with_msg:
	/* out_msg is global, so make sure we've zeroed it out */
	memset(&out_msg.data, 0, sizeof(out_msg.data));

	return ret;
}


static void bcast_status(void)
{
	struct stars_client *tmp, *c;
	LL_FOREACH_SAFE(clients, c, tmp) {
		if (!has_registered(c)) continue;

		out_msg.type = scheduler_on ? SCHED_ON : SCHED_OFF;
		if (send_message(c, &out_msg) < 0)
			delete_client(c);
	}
}


/*
 * Send a given message to a given client.
 *
 * We are particularly strict and consider the client dead if we encounter any
 * (even possibly recoverable if we were more lenient) error.
 */
static int send_message(struct stars_client *client, struct reqmessage *msg_p)
{
	ssize_t ret;
	char id_str[HEX_STR_LEN(client->id)];

	client_id_as_string(id_str, sizeof(id_str), client->id);

	ret = stars_send_noblock(client->fd, msg_p, sizeof(*msg_p));

	if (ret >= 0 && (size_t)ret < sizeof(*msg_p))/* Partial send */
		return -1;
	else if (ret < 0) {
		if (errno == EAGAIN ||
		    errno == EWOULDBLOCK ||
		    errno == ECONNRESET ||
		    errno == EPIPE) { /* Recoverable errors, but we're strict */
			log_info("Failed to send message to client %s",
				 id_str);
			return -1;
		} else log_fatal("stars_send_noblock() failed unrecoverably");
	} else { /* ret == 0 */
		log_info("Sent %s to client %s",
		         message_type_string[msg_p->type], id_str);
	}
	return 0;
}

/*
 * Send a given message to a given client.
 *
 * We are particularly strict and consider the client dead if we encounter any
 * (even possibly recoverable if we were more lenient) error.
 */
static int receive_message(struct stars_client *client, struct reqmessage *msg_p)
{
	ssize_t ret;
	char id_str[HEX_STR_LEN(client->id)];

	client_id_as_string(id_str, sizeof(id_str), client->id);

	ret = stars_receive_noblock(client->fd, msg_p, sizeof(*msg_p));

	if (ret == 0) { /* Client closed the other end of the connection */
		errno = ENOTCONN;
		log_debug("Client %s has closed the connection", id_str);
		return -1;
	} else if (ret > 0 && (size_t)ret < sizeof(*msg_p)) { /* Partial receive */
		return -1;
	} else if (ret < 0) {
		if (errno == EAGAIN ||
		    errno == EWOULDBLOCK ||
		    errno == ECONNRESET ||
		    errno == EPIPE) {
			log_info("Failed to receive message from client %s", id_str);
			return -1;
		} else log_fatal("stars_receive_noblock() failed unrecoverably");
	}
	return 0;
}


// A client sends REQ_LOCK when its model has a new task to execute.
static int FIFO_schedule(Task new_task)
{
	/**
	 * 0: A task is running without preemption.
	 * 1: The running task has been preempted.
	 * -1: The waiting queue is empty, either at startup or after all tasks finish.
	 * 2: Invalid state. A release request should already have updated the current task to NULL or the queue head.
	 * */


	log_debug("try_preemt_schedule  --- current_task state %d", current_task->state);
	if (current_task->state == 0)
	{
		if (task_queue_size == 0){
				log_debug("try_preemt_schedule() called with no pending tasks");
				*current_task = new_task;
				return -1;
			}
			else{
				return 2;
			}

	}else{
		return 1;
	}

}




/*
 * The timer thread's sole responsibility is to implement the Time Quantum (TQ)
 * notion of stars.
 *
 * When a client obtains the GPU lock, the timer resets.
 *
 * When TQ elapses, it sends a DROP_LOCK message to the client that holds the
 * lock. Send DROP_LOCK at the end of a fixed time quantum.
 */

void *timer_thr_fn(void *arg __attribute__((unused)))
{
	struct reqmessage t_msg = {0};
	unsigned int round_at_start;
	struct timespec timer_end_ts = {0, 0};
	int ret;
	int drop_lock_sent = 0;

	t_msg.id = 1337; /* Nobody checks this */
	t_msg.type = DROP_LOCK;

	true_or_exit(pthread_mutex_lock(&global_mutex) == 0);
	while (1) {
		must_reset_timer = 0;
		round_at_start = scheduling_round;
		true_or_exit(clock_gettime(CLOCK_REALTIME, &timer_end_ts) == 0);
		timer_end_ts.tv_sec += tq;
remainder:
		ret = pthread_cond_timedwait(&timer_cv, &global_mutex, &timer_end_ts);
		/* Wake up with global_mutex held, can do whatever we want */
		if (ret == ETIMEDOUT) { /* TQ elapsed */
			log_debug("TQ elapsed");
			if (!lock_held) continue; /* Life is meaningless :( */
			if (drop_lock_sent) continue; /* Send it only once */
			/*
			 * We use round_at_stat and scheduling_round to enable
			 * us to uniquely order (and by extent identify) every
			 * binding of the GPU lock to a client.
			 *
			 * This ensures we can avoid race conditions in which
			 * the timer wakes up and the lock has changed hands,
			 * thus erroneously sending a DROP_LOCK to the wrong
			 * client.
			 */
			if (round_at_start != scheduling_round) {
				drop_lock_sent = 0;
				continue;
			}
			/*
			 * Strict handling of clients. If something goes wrong,
			 * clean them up.
			 */
			if (send_message(current_task->client, &t_msg) < 0) {
				delete_client(current_task->client);
				drop_lock_sent = 0;
			} else { /* All good */
				drop_lock_sent = 1;
			}
		} else if (ret != 0) { /* Unrecoverable error */
			errno = ret;
			log_fatal("pthread_cond_timedwait()");
		} else { /* ret == 0, someone signaled the condvar */
			if (must_reset_timer) {
				drop_lock_sent = 0;
				continue;
			} else { /* Spurious wakeup */
				goto remainder;
			}
		}
	}
}



static void process_msg_v2(struct stars_client *client, const struct reqmessage *in_msg)
{
	int newtq;
	char id_str[HEX_STR_LEN(client->id)];
	char *endptr;

	// Process request metadata, including workload size, kernel block count, and SLO.
	client_id_as_string(id_str, sizeof(id_str), client->id);

	switch (in_msg->type) {
	case REGISTER:
		log_info("Received %s",
			   message_type_string[in_msg->type]);

		if (register_client(client, in_msg) < 0) delete_client(client);
		else log_info("Registered client %016" PRIx64 " with Pod"
			      " name = %s, Pod namespace = %s", client->id,
			      client->pod_name, client->pod_namespace);
		break;

	case SCHED_ON: /* starsctl */
		log_info("Received %s from %s",
                         message_type_string[in_msg->type], id_str);

		/*
		 * Ensure status actually changed before broadcasting,
		 * otherwise it is a no-op.
		 */
		if (!scheduler_on) {
			scheduler_on = 1;
			log_info("Scheduler turned ON, broadcasting it...");
			bcast_status();
		}
		break;

	case SCHED_OFF: /* starsctl */
		log_info("Received %s from %s",
			 message_type_string[in_msg->type], id_str);

		if (scheduler_on) {
			log_info("Scheduler turned OFF, broadcasting it...");
			scheduler_on = 0;
			bcast_status();
			/*
			 * When the scheduler is OFF, every client thinks they
			 * have the lock, so the requests list instantaneously
			 * becomes invalid. Empty it.
			 */
			struct stars_request *tmp, *r;
			LL_FOREACH_SAFE(requests, r, tmp) {
				LL_DELETE(requests, r);
				free(r);
			}
			lock_held = 0;
		}
		break;

	case SET_TQ: /* starsctl */
		log_info("Received %s from %s",
			 message_type_string[in_msg->type], id_str);

		errno = 0;
		newtq = (int)strtoll(in_msg->data, &endptr, 0);
                if (in_msg->data != endptr && *endptr == '\0' && errno == 0) {
			tq = newtq;
			must_reset_timer = 1;
			pthread_cond_broadcast(&timer_cv); /* Reset timer on TQ change */
			log_info("New TQ = %d", tq);
		}
		else log_info("Failed to parse new TQ from message");
		break;

	case REQ_LOCK: /* client */
		log_info("Received %s from %s",
			 message_type_string[in_msg->type], id_str);

		int state = 1;   // The task is active
		// KernelInfo *ks = in_msg.ks;


		char signature[256];
		double urgent_level = 0.0;
		int allsmcount = 0;
		double all_throughput = 0.0;

		int KS_L = scheduler(&actsk, in_msg->all_ks_num, actsk.submit_time, actsk.slo, 0);


		// Collect colocated-task information
		double predict_duration_part = insert_actask(actsk, &urgent_level, &allsmcount, &all_throughput, signature);

		int SM_POOL[MAX_SM_POOL_SIZE];
		int solo_duratoin_ns[MAX_SM_POOL_SIZE];
		memcpy(SM_POOL, in_msg->SM_POOL, sizeof(SM_POOL));
		memcpy(solo_duratoin_ns, in_msg->solo_duratoin_ns, sizeof(solo_duratoin_ns));
		int g_sm_pool_count = in_msg->g_sm_pool_count;
		int exp_sm = in_msg->exp_sm;

		if (has_registered(client)) {
			if (scheduler_on) {
				out_msg.type = LOCK_OK;
				out_msg.client_num = active_task_count;
				out_msg.KS_L = KS_L;
				out_msg.urgent_level = urgent_level;
				out_msg.all_smcount = allsmcount;
				out_msg.all_throughput = all_throughput;
				out_msg.duration_param = predict_duration_part;
				out_msg.select_sm = exp_sm;
				memcpy(out_msg.signature, signature, sizeof(out_msg.signature));

				int ret1 = send_message(client, &out_msg);
			}
		} else { /* The client is not registered. Slam the door. */
			delete_client(client);
		}



		break;

	case LOCK_RELEASED: /* From client */
		log_info("Received %s from %s",
			 message_type_string[in_msg->type], id_str);

		if (has_registered(client)) {
			/*
			 * When the scheduler is OFF, LOCK_RELEASED messages
			 * are meaningless. Mostly a sanity check.
			 */
			if (scheduler_on) {
				if (release_actask(client)){
					out_msg.type = LOCK_RELEASED;
					int ret_ok = send_message(client, &out_msg);
					break;
				} else {
					lock_held = 0;
				}
			}
		} else { /* The client is not registered. Slam the door. */
			delete_client(client);
		}
		break;


	default: /* Unknown message type */
		log_info("Received message of unknown type %d"
			 " from %s", (int)in_msg->type, id_str);
		break;
	}
}

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused)))
{
	pthread_t timer_tid;
	pthread_t insert_tid;
	pthread_t pop_tid;
	struct stars_client *client = NULL;
	int ret, err, lsock, rsock, num_fds;
	char *debug_val;
	char *weight_val;
	char *delta_val;
	struct reqmessage in_msg = {0};
	struct epoll_event event, events[EPOLL_MAX_EVENTS];

	double z = 0.0;
	Task init_task = {0, client, z, z, z, z, z, z, z, z};
	current_task  = &init_task;


	Task init_in_task = {0, client, z, z, z, z, z, z, z, z};
	in_task = &init_in_task;


	debug_val = getenv(ENV_STARS_DEBUG);
	if (debug_val != NULL) {
		__debug = 1;
		log_info("stars-scheduler started in debug mode");
	} else log_info("stars-scheduler started in normal mode");

	weight_val = getenv("STARS_W_SFR");
	if (weight_val != NULL) {
		char *weight_end = NULL;
		errno = 0;
		double parsed_weight = strtod(weight_val, &weight_end);
		if (errno != 0 || weight_end == weight_val || *weight_end != '\0' ||
			!isfinite(parsed_weight) || parsed_weight < 0.0 || parsed_weight > 1.0) {
			log_fatal("Invalid STARS_W_SFR='%s'; expected a number in [0,1]", weight_val);
		}
		W_sfr = parsed_weight;
	}
	log_info("Temporal risk weights: W_sfr=%.3f, W_prr=%.3f", W_sfr, 1.0 - W_sfr);

	delta_val = getenv("STARS_RISK_DELTA");
	if (delta_val != NULL) {
		char *delta_end = NULL;
		errno = 0;
		double parsed_delta = strtod(delta_val, &delta_end);
		if (errno != 0 || delta_end == delta_val || *delta_end != '\0' ||
			!isfinite(parsed_delta) || parsed_delta <= 0.0 || parsed_delta >= 1.0) {
			log_fatal("Invalid STARS_RISK_DELTA='%s'; expected a number in (0,1)", delta_val);
		}
		risk_delta = parsed_delta;
	}
	log_info("Temporal risk deadband: delta=%.3f, bounds=[%.3f, %.3f]",
		risk_delta, 1.0 - risk_delta, 1.0 + risk_delta);

	/*
	 * Permissions are 711:
	 * - RWX (7) for owner (root because we are under /var/run/)
	 * - X (1) for group (to connect to the socket)
	 * - X (1) for others (to connect to the socket)
	 *
	 * In general, directory permissions work as follows:
	 *
	 * - `r`: Can read the names of files in the directory
	 * - `w`:
	 *    + if also `x`: can create, rename, delete files in the directory
	 *    + if not `x`: nothing
	 * - `x`:
	 *    1. Can access files (read/write to them if individual file
	 *       permissions permit) in the directory
	 *    2. Can read meta-information (permissions, timestamps, size)
	 *       about a file if the file name is known
	 */
	err = mkdir(STARS_SOCK_DIR, S_IRWXU | S_IXGRP | S_IXOTH);
	if (err != 0 && errno != EEXIST)
		log_fatal("Could not create scheduler socket directory %s",
			  STARS_SOCK_DIR);

	/*
	 * Unconditionally call chmod(), as it is not affected by umask, to
	 * ensure that the directory has the correct (aforementioned) 711
	 * permissions.
	 */
	if (chmod(STARS_SOCK_DIR, S_IRWXU | S_IXGRP | S_IXOTH) != 0)
		log_fatal("chmod() failed for %s", STARS_SOCK_DIR);

	/* TODO: Enable setting this dynamically through an envvar/conffile */
	scheduler_on = 1;
	/* TODO: Enable setting this dynamically through an envvar/conffile */
	tq = STARS_DEFAULT_TQ;

	/* Seed srand() for generating client IDs */
	srand((unsigned int)(time(NULL)));

	true_or_exit(pthread_mutex_init(&global_mutex, NULL) == 0);
	true_or_exit(pthread_cond_init(&timer_cv, NULL) == 0);

	if (stars_get_scheduler_path(nvscheduler_socket_path) != 0)
		log_fatal("stars_get_scheduler_path() failed!");

	/* Spawn the timer thread */
	// Legacy time-quantum thread: retained but disabled for this scheduler
	// true_or_exit(pthread_create(&timer_tid, NULL, timer_thr_fn, NULL) == 0);

	// Queue-insertion thread, separate from normal request handling
	// true_or_exit(pthread_create(&insert_tid, NULL, DP_insert_task, NULL) == 0);

	/* Set up fd for epoll */
	true_or_exit((epoll_fd = epoll_create(1)) >= 0);

	/* Start listening */
	true_or_exit(stars_bind_and_listen(&lsock, nvscheduler_socket_path) == 0);


	/* Use the 'fd' field of epoll_data for the listening socket events */
	event.data.fd = lsock;
	event.events = EPOLLIN;
	true_or_exit(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, lsock, &event) == 0);

	/*
	 * According to man unix(7):
	 * On Linux, connecting to a stream socket object requires write
	 * permission on that socket; [...]
	 *
	 * We also need execute permission for the socket directory, to access
	 * the socket file that lies therein.
	 *
	 * Therefore, the minimal permissions for the socket file are 722.
	 */
	if (chmod(nvscheduler_socket_path, S_IRWXU | S_IWGRP| S_IWOTH) != 0)
		log_fatal("chmod() failed for %s", nvscheduler_socket_path);

	out_msg.id = 7331;
	out_msg.proc_time = 0.0;
	out_msg.slo = 0.0;
	out_msg.client_num = 0;

	log_info("stars-scheduler listening on %s",
		 nvscheduler_socket_path);

	for (;;) {
		num_fds = RETRY_INTR(epoll_wait(epoll_fd, events, EPOLL_MAX_EVENTS, -1));

		/* Since we use an infinite timeout, a non-zero return value
		 * indicates an error
		 */
		if (num_fds < 0) log_fatal("epoll_wait() failed");

		/* ret > 0, we got an event */

		true_or_exit(pthread_mutex_lock(&global_mutex) == 0);

		for (int i = 0; i < num_fds; i++) {
			if (events[i].data.fd == lsock) {
				/* New connection. */
				ret = stars_accept(events[i].data.fd, &rsock);
				if (ret == 0) { /* OK */
					/* 1. Set up the client struct */
					client = malloc(sizeof(*client));
					client->fd = rsock;
					client->id = STARS_UNREGISTERED_ID;
					client->next = NULL;

					/*
					 * 2. Add new rsock to the epoll
					 *    interest list.
					 *
					 * The epoll_data field of epoll_event
					 * is a union, and we can only use one
					 * field at a time.
					 *
					 * In this case, use void *ptr.
					 *
					 * See man epoll_ctl(2).
					 * */
					event.data.ptr = client;
					event.events = EPOLLIN;
					if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, rsock, &event) < 0) {
						log_warn("Couldn't add %d to the epoll interest list", rsock);
						log_debug("Couldn't add %d to the epoll interest list", rsock);
						close(rsock);
						free(client);
					} else LL_APPEND(clients, client); /* OK */
				} else if (errno != ECONNABORTED && errno != EAGAIN && errno != EWOULDBLOCK)
					log_fatal("accept() failed non-transiently");

			} else { /* Some event other than new connection */
				client = (struct stars_client *)events[i].data.ptr;

				/* Check for incoming messages */
				if (events[i].events & EPOLLIN) {
					ret = receive_message(client, &in_msg);
					if (ret < 0) {
						delete_client(client);
					}
					else process_msg_v2(client, &in_msg); /* OK */
					// log_debug("after process_msg_v2");
				}
				/*
				 * Check for errors after checking for messages,
				 * as starsctl sends a message and immediately
				 * closes a connection.
				 */
				else if (events[i].events & (EPOLLERR | EPOLLHUP)) {
					delete_client(client);
				}

			}
		}
		true_or_exit(pthread_mutex_unlock(&global_mutex) == 0);
		// log_debug("after pthread_mutex_unlock");
	}

	/* Control should never reach here */
	return -1;
}
