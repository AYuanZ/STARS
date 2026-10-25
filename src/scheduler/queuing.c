#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "queuing.h"
#include "common.h"
// Maximum number of tasks
#define MAX_TASKS 20
#define MAX_QUEUE_SIZE 100
#define HEX_STR_LEN(x) (2 * sizeof(x) + 1)

static void client_id_as_string(char *buf, size_t buflen, uint64_t id)
{
	if (id == STARS_UNREGISTERED_ID)
		strlcpy(buf, "<UNREGISTERED>", buflen);
	else snprintf(buf, buflen, "%016" PRIx64, id);
}


// Function to calculate sum of processing times
double calculate_p(Task* tasks, int num_tasks) {
    double sum = 0.0;
    for(int i = 0; i < num_tasks; i++)
        sum += tasks[i].processing_time;
    return sum;
}

// Function to print tasks (for debugging)
void print_tasks(Task* tasks, int num_tasks) {
    log_debug("[");
    for(int i = 0; i < num_tasks; i++) {
        char id_str[HEX_STR_LEN(tasks[i].client->id)];
        client_id_as_string(id_str, sizeof(id_str), tasks[i].client->id);
        log_debug("{ID: %s, RT: %.2lf, DL: %.2lf, PT: %.2lf, SLO: %.3lf, W: %.3lf}",
            id_str,
            tasks[i].release_time, tasks[i].deadline, tasks[i].processing_time,
            tasks[i].slo, tasks[i].weight);
        // if(i < num_tasks -1)
            // log_debug(", ");
        // log_debug("\n");
    }
    printf("]\n");
}

// Function to perform weighted tardiness dynamic programming
double weighted_tardiness(Task* tasks, int n, int* optimal_size, Task** optimal_sequence)
{
    if(n > 20) { // limit to 20 tasks
        printf("Too many tasks\n");
        exit(1);
    }

    int total_subsets = 1 << n;

    double* f = (double*)malloc(sizeof(double) * total_subsets);
    double* C = (double*)malloc(sizeof(double) * total_subsets);

    // Initialize all f to DBL_MAX
    for(int s = 0; s < total_subsets; s++) {
        f[s] = DBL_MAX;
        C[s] = 0.0;
    }

    // Base case: empty set
    f[0] = 0.0;
    C[0] = 0.0;

    // Iterate through all subsets
    for(int s = 1; s < total_subsets; s++) {
        // Iterate through tasks to find which are in the subset
        for(int t = 0; t < n; t++) {
            if(s & (1 << t)) { // if task t is in subset
                int prev_s = s ^ (1 << t); // subset without task t
                double completion_time = C[prev_s] + tasks[t].processing_time;
                double com = completion_time + tasks[t].release_time;
                double tardiness = (com > tasks[t].deadline) ? (com - tasks[t].deadline) : 0.0;
                double weighted_tardiness_val = tasks[t].weight * tardiness;
                double f_current = f[prev_s] + weighted_tardiness_val;

                if(f_current < f[s]) {
                    f[s] = f_current;
                    C[s] = completion_time;
                }
            }
        }
    }

    // Optimal tardiness is for the full set
    double optimal_tardiness = f[total_subsets -1];

    // Backtrace to find the optimal sequence
    Task* sequence = (Task*)malloc(sizeof(Task) * n);
    int seq_size = 0;
    int current_s = total_subsets -1;

    while(current_s != 0) {
        for(int t = 0; t < n; t++) {
            if(current_s & (1 << t)) {
                int prev_s = current_s ^ (1 << t);
                double completion_time = C[prev_s] + tasks[t].processing_time + tasks[t].release_time;
                double tardiness = (completion_time > tasks[t].deadline) ? (completion_time - tasks[t].deadline) : 0.0;
                double f_current = f[prev_s] + (tasks[t].weight * tardiness);
                if(fabs(f_current - f[current_s]) < 1e-6) { // floating point comparison
                    sequence[seq_size++] = tasks[t];
                    current_s = prev_s;
                    break;
                }
            }
        }
    }

    // Reverse the sequence
    for(int i = 0; i < seq_size /2; i++) {
        Task temp = sequence[i];
        sequence[i] = sequence[seq_size -1 -i];
        sequence[seq_size -1 -i] = temp;
    }

    // Assign outputs
    *optimal_size = seq_size;
    *optimal_sequence = sequence;

    // Free memory
    free(f);
    free(C);

    return optimal_tardiness;
}

// Function to determine optimal sequence with new task
void optimal_sequence_with_new_task(Task* Q, int Q_size, Task new_task, Task** B, int* B_size, Task** A, int* A_size) {
    // Initialize J as Q + new_task
    int J_size = Q_size + 1;
    Task* J = (Task*)malloc(sizeof(Task) * J_size);
    for(int i = 0; i < Q_size; i++)
        J[i] = Q[i];
    J[Q_size] = new_task;

    int i = J_size - 2; // len(J) - 2

    while (i >= 0) {
        Task j_task = J[i+1];
        Task k_task = J[i];

        // Compute P(J[:i]) = sum of processing_time of J[0 to i-1]
        double P_J_i = calculate_p(J, i); // sum of processing_time up to i (exclusive)

        // Lemma 1: j.processing_time <= k.processing_time AND j.weight >= k.weight AND j.deadline <= max(k.deadline, k.release_time + P(J[:i]) + k.processing_time)
        double max_dk = fmax(k_task.deadline, k_task.release_time + P_J_i + k_task.processing_time);
        if (j_task.processing_time <= k_task.processing_time &&
            j_task.weight >= k_task.weight &&
            j_task.deadline <= max_dk) {
            // Swap J[i] and J[i+1]
            Task temp = J[i];
            J[i] = J[i+1];
            J[i+1] = temp;
            i--;
            continue;
        }

        // Lemma 2: j.weight >= k.weight AND k.deadline >= max(j.deadline, k.release_time + P(J[:i+2]) - k.processing_time)
        double P_J_i_plus2 = calculate_p(J, i+2); // sum up to i+1
        double max_d = fmax(j_task.deadline, k_task.release_time + P_J_i_plus2 - k_task.processing_time);
        if (j_task.weight >= k_task.weight &&
            k_task.deadline >= max_d) {
            // Swap J[i] and J[i+1]
            Task temp = J[i];
            J[i] = J[i+1];
            J[i+1] = temp;
            i--;
            continue;
        }

        // Lemma 3: k.deadline >= P(J[:i+2])
        if (k_task.deadline >= (k_task.release_time + calculate_p(J, i+2))) {
            // Swap J[i] and J[i+1]
            Task temp = J[i];
            J[i] = J[i+1];
            J[i+1] = temp;
            i--;
            continue;
        }

        // Else, break
        break;
    }

    // Set B = J[0 to i+1]
    *B_size = i + 2;
    *B = (Task*)malloc(sizeof(Task) * (*B_size));
    for(int idx = 0; idx < *B_size; idx++)
        (*B)[idx] = J[idx];

    // Set A = J[i+2 to end]
    *A_size = J_size - (*B_size);
    if(*A_size > 0) {
        *A = (Task*)malloc(sizeof(Task) * (*A_size));
        for(int idx = 0; idx < *A_size; idx++)
            (*A)[idx] = J[(*B_size) + idx];
    }
    else {
        *A = NULL;
    }

    free(J);
}

// Function to perform dynamic programming algorithm
void dp_algorithm(Task insert_task, Task* waiting_queue, int waiting_size, int* final_size) {
    Task* final_optimal_queue = NULL;
    if (waiting_size == 0) {
        // If waiting queue is empty, return insert_task
        // final_optimal_queue = (Task*)malloc(sizeof(Task));
        // final_optimal_queue[0] = insert_task;
        waiting_queue[0] = insert_task;
        *final_size = 1;
        print_tasks(waiting_queue, *final_size);
    }
    else {
        // Update remain_slo and weight for each task in waiting_queue
        for(int i = 0; i < waiting_size; i++) {
            Task* w_task = &waiting_queue[i];
            w_task->release_time = insert_task.release_time;
            char id_str[HEX_STR_LEN(w_task->client->id)];

            if ((w_task->deadline - w_task->release_time) <= 0) {
                w_task->remain_slo = 0.001 * w_task->processing_time;
            }
            else {
                w_task->remain_slo = w_task->deadline - w_task->release_time;
            }
            // Update release_time

            // Update weight
            w_task->weight = 1.0 / w_task->remain_slo;

        }

        // Update insert_task
         if ((insert_task.deadline - insert_task.release_time) <= 0) {
            insert_task.remain_slo = 0.001 * insert_task.processing_time;
        }
        else {
            insert_task.remain_slo = insert_task.deadline - insert_task.release_time;
        }
        // insert_task.remain_slo = fmax(0.001, insert_task.deadline - insert_task.release_time);
        insert_task.weight = 1.0 / insert_task.remain_slo;
        // Get B and A from optimal_sequence_with_new_task
        Task* B = NULL;
        int B_size = 0;
        Task* A = NULL;
        int A_size = 0;
        optimal_sequence_with_new_task(waiting_queue, waiting_size, insert_task, &B, &B_size, &A, &A_size);

        Task* optimal_B = NULL;
        int optimal_B_size = 0;
        if (B_size == 1) {
            // If B has length 1, optimal_B = B
            optimal_B = B;
            optimal_B_size = B_size;
        }
        else {
            // Else, perform weighted_tardiness on B
            int opt_size = 0;
            Task* optimal_sequence = NULL;
            double tardiness = weighted_tardiness(B, B_size, &opt_size, &optimal_sequence);
            optimal_B = optimal_sequence;
            optimal_B_size = opt_size;
            free(B);
        }

        // Combine optimal_B and A to form final_optimal_queue
        // final_optimal_queue = (Task*)malloc(sizeof(Task) * (optimal_B_size + A_size));
        for(int i = 0; i < optimal_B_size; i++)
            waiting_queue[i] = optimal_B[i];
        for(int i = 0; i < A_size; i++)
            waiting_queue[optimal_B_size + i] = A[i];
        *final_size = optimal_B_size + A_size;

        print_tasks(waiting_queue, *final_size);

        free(optimal_B);
        free(A);
    }


}

// First-in, first-out task ordering
void FIFO(Actask insert_task, Actask* waiting_queue, int* queue_size) {
    waiting_queue[(*queue_size)] = insert_task;
    (*queue_size)++;

    // print_tasks(waiting_queue, *queue_size);
}

/* The temporal FIFO baseline queues Task records, while the MPS scheduler
 * tracks Actask records. Keep their storage layouts separate. */
void FIFO_task(Task insert_task, Task* waiting_queue, int* queue_size) {
    waiting_queue[(*queue_size)++] = insert_task;
}

// Earliest-deadline-first ordering
void EDF(Task insert_task, Task* waiting_queue, int* queue_size) {
    // Append the new task
    waiting_queue[(*queue_size)] = insert_task;
    (*queue_size)++;

    // Sort the queue based on earliest deadline first
    for(int i = (*queue_size) - 1; i > 0; i--) {
        if(waiting_queue[i].deadline < waiting_queue[i - 1].deadline) {
            // Swap
            Task temp = waiting_queue[i];
            waiting_queue[i] = waiting_queue[i - 1];
            waiting_queue[i - 1] = temp;
        } else {
            break;
        }
    }
}

void SJF(Task insert_task, Task* waiting_queue, int* queue_size) {
    // Append the new task
    waiting_queue[(*queue_size)] = insert_task;
    (*queue_size)++;

    // Sort the queue based on earliest deadline first
    for(int i = (*queue_size) - 1; i > 0; i--) {
        if(waiting_queue[i].processing_time < waiting_queue[i - 1].processing_time) {
            // Swap
            Task temp = waiting_queue[i];
            waiting_queue[i] = waiting_queue[i - 1];
            waiting_queue[i - 1] = temp;
        } else {
            break;
        }
    }
}


void SRTF(Task insert_task, Task* waiting_queue, int *queue_size) {
    // if(queue_size >= MAX_QUEUE_SIZE) {
    //     printf("Queue is full!\n");
    //     return queue_size;
    // }

    // Append the new task
    waiting_queue[(*queue_size)] = insert_task;
    (*queue_size)++;

    // Sort the queue based on shortest remaining time first
    for(int i = (*queue_size) - 1; i > 0; i--) {
        if(waiting_queue[i].remaining_time <= waiting_queue[i - 1].remaining_time) {
            // Swap
            Task temp = waiting_queue[i];
            waiting_queue[i] = waiting_queue[i - 1];
            waiting_queue[i - 1] = temp;
        } else {
            break;
        }
    }
    print_tasks(waiting_queue, *queue_size);
}

void HTW(Task insert_task, Task* waiting_queue, int* queue_size) {

    // Append the new task
    waiting_queue[(*queue_size)] = insert_task;
    (*queue_size)++;

    // Update release_time for all tasks
    for(int i = 0; i < (*queue_size); i++) {
        waiting_queue[i].release_time = insert_task.release_time;
    }

    // Sort based on decisive_ratio_new >= decisive_ratio_run
    for(int i = (*queue_size) - 1; i > 0; i--) {
        Task j = waiting_queue[i];
        Task k = waiting_queue[i - 1];

        double rat_new = fmax(0, j.deadline - j.release_time - k.lat_prmp);
        double rat_run = fmax(0, k.deadline - j.release_time - k.lat_prmp);

        double rpt_new = fmax(0.0001, j.remaining_time);
        double rpt_run = fmax(0.00001, (k.remaining_time - k.lat_prmp));

        double decisive_ratio_new = 1.0 / (j.slo * fmax(rpt_new, rat_new));
        double decisive_ratio_run = 1.0 / (k.slo * fmax(rpt_run, rat_run));

        if(decisive_ratio_new >= decisive_ratio_run) {
            // Swap
            Task temp = waiting_queue[i];
            waiting_queue[i] = waiting_queue[i - 1];
            waiting_queue[i - 1] = temp;
        } else {
            break;
        }
    }

}






// Example usage
// int main() {
//     // Example tasks
//     // Initialize waiting_queue with some tasks
//     Task waiting_queue[MAX_QUEUE_SIZE];
//     int queue_size = 0;


// //     // Task structure
// // typedef struct {
// //     int id; // Unique identifier for debugging
// //     double release_time;
// //     double processing_time;
// //     double slo;       // Service Level Objective
// //     double remain_slo;
// //     double deadline;
// //     double remaining_time;
// //     double lat_prmp; // Latency or other parameter as per your logic
// //     double weight;
// // } Task;
//     double a = 1.0;
//     double b = 2.0;
//     double c = (a + b);
//     printf("%f \n", c);

//     Task task1 = {1, 0.0, 3.0, 5.0, 5.0, 5.0, 3.0, 0.05, (1 / 5.0)};
//     Task task2 = {2, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.05, (1 / 2.0)};
//     Task task3 = {3, 0.0, 1.0, 3.0, 3.0, 3.0, 1.0, 0.05, (1 / 3.0)};
//     Task task4 = {4, 0.0, 4.0, 10.0, 10.0, 10.0, 4.0, 0.05, (1 / 10.0)};
//     Task task5 = {5, 0.0, 3.0, 4.0, 4.0, 4.0, 3.0, 0.05, (1 / 4.0)};
//     Task task6 = {6, 0.0, 2.0, 5.0, 5.0, 5.0, 2.0, 0.05, (1 / 5.0)};
//     Task new_task = {12, 2.0, 2.0, 2.0, 0.001, 1.0, 2.0, 0.05, (1 / 0.001)};


//     // Perform dp_algorithm
//     int final_size = 0;
//     // Task* final_queue = dp_algorithm(insert_task,    , waiting_size, &final_size);

//     Task final_queue[MAX_QUEUE_SIZE];
//     printf("Dynamic Programming Scheduling:\n");
//     dp_algorithm(task1, final_queue, final_size, &final_size);
//     print_tasks(final_queue, final_size);
//     dp_algorithm(task2, final_queue, final_size, &final_size);
//     print_tasks(final_queue, final_size);
//     dp_algorithm(task3, final_queue, final_size, &final_size);
//     print_tasks(final_queue, final_size);
//     dp_algorithm(task4, final_queue, final_size, &final_size);
//     print_tasks(final_queue, final_size);
//     dp_algorithm(task5, final_queue, final_size, &final_size);
//     print_tasks(final_queue, final_size);
//     dp_algorithm(task6, final_queue, final_size, &final_size);
//     print_tasks(final_queue, final_size);
//     dp_algorithm(new_task, final_queue, final_size, &final_size);

//     // print_list(final_queue, queue_size);
//     // calculate_weighted_tardiness(waiting_queue, queue_size);
//     printf("\n");



//     // Print final optimal queue
//     printf("Final Optimal Queue:\n");
//     print_tasks(final_queue, final_size);

//     // Free allocated memory
//     // free(final_queue);

//     return 0;
// }
