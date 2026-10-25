#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "comm.h"
#include "common.h"
#include "cinfo.h"
// Maximum number of tasks
#define MAX_TASKS 20
#define MAX_QUEUE_SIZE 100




typedef struct stars_client{
	int fd; /* server-side socket for the persistent connection */
	uint64_t id; /* Unique */
	char pod_name[POD_NAME_LEN_MAX];
	char pod_namespace[POD_NAMESPACE_LEN_MAX];
	struct stars_client *next;
} Client;


// Task structure
// typedef struct {z
//     int id; // Unique identifier for debugging
//     double release_time;
//     double processing_time;
//     double slo;       // Service Level Objective
//     double remain_slo;
//     double deadline;
//     double remaining_time;
//     double lat_prmp; // Latency or other parameter as per your logic
//     double weight;
// } Task;

typedef struct myshare_task
{
    int state; // 1 if this record contains an active task; 0 if it is empty
	/* data */
	Client *client;  // Client associated with the request
	double release_time;
    double processing_time;
    double slo;       // Service Level Objective
    double remain_slo;
    double deadline;
    double remaining_time;
    double lat_prmp; // Latency or other parameter as per your logic
    double weight;
} Task;


// MPS: information about an active (running) task
typedef struct active_task{
    Client *client;         // Client associated with this task
    KernelInfo ks;          // Kernel segment currently executed by this client
    int smcount;            // SM allocation used by this client
    int slo;             // Service Level Objective
    int all_ks_num;         // Total number of kernel segments for this task
    double submit_time;     // Time of the first submission to the scheduler
    double urgent_level;    // Task urgency
} Actask;


extern void dp_algorithm(Task insert_task, Task* waiting_queue, int waiting_size, int* final_size);
extern void FIFO(Actask insert_task, Actask* waiting_queue, int *queue_size);
extern void FIFO_task(Task insert_task, Task* waiting_queue, int *queue_size);
extern void EDF(Task insert_task, Task* waiting_queue, int *queue_size);
extern void SRTF(Task insert_task, Task* waiting_queue, int *queue_size);
extern void SJF(Task insert_task, Task* waiting_queue, int *queue_size);
extern void HTW(Task insert_task, Task* waiting_queue, int *queue_size);
