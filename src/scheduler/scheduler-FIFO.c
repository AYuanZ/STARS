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

#define STARS_DEFAULT_TQ 200000

#define MAX_QUEUE_SIZE 100 // Maximum number of tasks in the candidate queue

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



/* Holds the requests for the GPU lock, which we serve in an FCFS manner */
struct stars_request {
	struct stars_client *client;
	struct stars_request *next;
};


// Initialize the candidate queue and its size
Task waiting_task_queue[MAX_QUEUE_SIZE];
int task_queue_size = 0;
Task *current_task = NULL;   // Currently running task
Task *preemted_task = NULL;  // Preempted task
Task *in_task = NULL;       // Task entering the waiting queue


struct stars_client *clients = NULL;
struct stars_request *requests = NULL;  // List of lock requests

void *timer_thr_fn(void *arg __attribute__((unused)));

void *DP_insert_task(void *arg __attribute__((unused)));
void *pop_task(void *arg __attribute__((unused)));

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

// static void try_schedule_v2(void);
static void insert_task(Task in_task);
static int preempt_scheduling(Task current_task, Task new_task);
static int try_preemt_schedule(Task new_task);
static void try_schedule_release(void);
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


// Current timestamp in milliseconds
static double get_current_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Insert a task into the candidate queue
static void insert_task(Task in_task)
{
	FIFO_task(in_task, waiting_task_queue, &task_queue_size);

}

// Remove tasks belonging to a disconnected client
static void delete_task(struct stars_client *client)
{
    for (int i = 0; i < task_queue_size; i++)
	{
		if (client->fd == waiting_task_queue[i].client->fd)
		{
			 // Shift subsequent entries forward
            for (int j = i; j < task_queue_size - 1; j++)
			{
                waiting_task_queue[j] = waiting_task_queue[j + 1];
            }
			task_queue_size--;
		}
    }
}

// Remove tasks belonging to a disconnected client
static int release_task(struct stars_client *client)
{
    for (int i = 0; i < task_queue_size; ++i) {
        if (client->fd == waiting_task_queue[i].client->fd) {
            for (int j = i; j < task_queue_size - 1; ++j)
                waiting_task_queue[j] = waiting_task_queue[j + 1];
            --task_queue_size;
            return 1;
        }
    }
    return 0;
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
			log_info("Failed to receive message from client %s",
				 id_str);
			return -1;
		} else log_fatal("stars_receive_noblock() failed unrecoverably");
	}
	return 0;
}

// Decide whether to preempt the current task
static int preempt_scheduling(Task run_task, Task new_task1) {
    // Current timestamp in milliseconds
    double t_now = get_current_time_ms();

    // Compute the time remaining until the deadline
    double rat_new = fmax(0, new_task1.deadline - t_now - run_task.lat_prmp);
    double rat_run = fmax(0, run_task.deadline - t_now - run_task.lat_prmp);

    // Compute the remaining execution time
    double rpt_new = fmax(0.001, new_task1.remaining_time);
    // double rpt_run = fmax(0.001, run_task.remaining_time  - t_now + run_task.release_time - run_task.lat_prmp);
	double rpt_run = fmax(0.001, run_task.remaining_time  + run_task.release_time - t_now  - run_task.lat_prmp);

	// if (rpt_run == 0){
	// 	return 0;
	// }

    // Compute the scheduling decision ratio
    double decisive_ratio_new = 1 / (new_task1.slo * fmax(rpt_new, rat_new));
    double decisive_ratio_run = 1 / (run_task.slo * fmax(rpt_run, rat_run));
    // Compare decision ratios to decide whether to preempt
    if (decisive_ratio_new > decisive_ratio_run) {
        return 1; // The new task has higher priority; preempt
    } else {
        return 0; // The current task has higher priority; do not preempt
    }
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




/**
 * A completed task sends a release_lock request.
 * If the waiting queue is empty, clear the current task.
 * Otherwise, select the task at the head of the queue.
*/
static void try_schedule_release(void)
{
	if (task_queue_size == 0){
		log_debug("try_schedule_release task_queue_size == 0 task_queue_size = %d", task_queue_size);

		log_debug("try_schedule_release() called with no pending tasks");
		current_task->state = 0;

		log_debug("try_preemt_schedule  --- current_tasks state %d", current_task->state);
	}
	else{
		log_debug("try_schedule_release task_queue_size != 0 task_queue_size = %d", task_queue_size);
		lock_held = 1;
		out_msg.type = LOCK_OK;
		*current_task = waiting_task_queue[0];
		// pthread_cond_broadcast(&pop_cv);
		for (int i = 0; i < task_queue_size - 1; i++)
		{
			waiting_task_queue[i] = waiting_task_queue[i+1];
		}
		task_queue_size--;
		int ret_r = send_message(current_task->client, &out_msg);
		current_task->release_time = get_current_time_ms();
		char current_str[HEX_STR_LEN(current_task->client->id)];
		if (ret_r < 0) {
			log_debug("Send to current task err! outmsg type %d , client ID--%s",(int)out_msg.type, current_str);
			return ;
		}else{
			lock_held = 1;
		}
	}


	// log_debug("after try_schedule_release");

}




void *DP_insert_task(void *arg __attribute__((unused))){
	while (1)
	{
		true_or_exit(pthread_mutex_lock(&insert_mutex) == 0);
		int ret = pthread_cond_wait(&insert_cv, &insert_mutex);
		FIFO_task(*in_task, waiting_task_queue, &task_queue_size);
		true_or_exit(pthread_mutex_unlock(&insert_mutex) == 0);
	}
}

void *pop_task(void *arg __attribute__((unused))){
	while (1)
	{
		true_or_exit(pthread_mutex_lock(&insert_mutex) == 0);
		int ret = pthread_cond_wait(&pop_cv, &insert_mutex);
		for (int i = 0; i < task_queue_size; i++)
		{
			waiting_task_queue[i] = waiting_task_queue[i+1];
		}
		true_or_exit(pthread_mutex_unlock(&insert_mutex) == 0);
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
				try_schedule_release();
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
		double release_time = get_current_time_ms();
		int slo_init = (int)(in_msg->slo);
		int slo_xs = slo_init % 10;
		double slo = slo_init / 10;
		double processing_time = (slo / slo_xs); // Heuristic duration estimate; replace with a calibrated prediction
		double remain_slo = slo;
		double deadline = (release_time + slo);
		double remaining_time = processing_time;
		double lat_prmp = (processing_time > 20) ? 5 : (processing_time / 10);
		double weight = (1.0 / slo);

		log_debug("Current task SLO---------%f---------proc_time=%f", slo, processing_time);

		// Initialize the task
		Task new_task = {state, client, release_time, processing_time, slo, remain_slo, deadline, remaining_time, lat_prmp, weight};

		log_debug("process_msg_v2 REQ_LOCK----------");
		if (has_registered(client)) {
			if (scheduler_on) {
				int sch_cv = FIFO_schedule(new_task);
				log_debug("process_msg_v2 REQ_LOCK FIFO_schedule---------====%d", sch_cv);
				switch (sch_cv) {
				case -1:
					out_msg.type = LOCK_OK;
					int ret1 = send_message(current_task->client, &out_msg);
					current_task->release_time = get_current_time_ms();
					lock_held = 1;
					break;
				case 1:
					*in_task = new_task;
					true_or_exit(pthread_mutex_lock(&insert_mutex) == 0);
					FIFO_task(*in_task, waiting_task_queue, &task_queue_size);
					true_or_exit(pthread_mutex_unlock(&insert_mutex) == 0);
				}
			}
		} else { /* The client is not registered. Slam the door. */
			delete_client(client);
			delete_task(client);
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
				if (release_task(client)){
					out_msg.type = LOCK_RELEASED;
					int ret_ok = send_message(client, &out_msg);
					break;
				} else {
					lock_held = 0;
					try_schedule_release();
				}
			}
		} else { /* The client is not registered. Slam the door. */
			delete_client(client);
		}
		// log_debug("after try_schedule_release");
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
	struct reqmessage in_msg = {0};
	struct epoll_event event, events[EPOLL_MAX_EVENTS];

	double z = 0.0;
	Task init_task = {0, client, z, z, z, z, z, z, z, z};
	current_task  = &init_task;

	Task init_prmt_task = {0, client, z, z, z, z, z, z, z, z};
	preemted_task = &init_prmt_task;

	Task init_in_task = {0, client, z, z, z, z, z, z, z, z};
	in_task = &init_in_task;


	debug_val = getenv(ENV_STARS_DEBUG);
	if (debug_val != NULL) {
		__debug = 1;
		log_info("stars-scheduler started in debug mode");
	} else log_info("stars-scheduler started in normal mode");

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
	true_or_exit(pthread_create(&insert_tid, NULL, DP_insert_task, NULL) == 0);
	true_or_exit(pthread_create(&pop_tid, NULL, pop_task, NULL) == 0);

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
						if (!lock_held && scheduler_on) try_schedule_release();
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
					if (!lock_held && scheduler_on)
						try_schedule_release();
				}

			}
		}
		true_or_exit(pthread_mutex_unlock(&global_mutex) == 0);
		// log_debug("after pthread_mutex_unlock");
	}

	/* Control should never reach here */
	return -1;
}
