#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CINFO_H_
#define CINFO_H_
// Kernel-segment attributes for different SM allocations
#define HASH_TABLE_SIZE 9983 // Choose a prime appropriate for the dataset size
#define MAX_LINE_LEN 256

// One row of the kernel-profile CSV
typedef struct {
    int ks_id;
    int kernel_num;
    int duration_ns;
    double sm_usage;
    double dram_demand;
    double l2_demand;
    int now_duration_ns; // Ideal cumulative duration through the current kernel segment
    int solo_duration_ns; // Standalone duration of the current request
} KernelInfo;

// Hash-table node containing a record and a link to the next node
typedef struct HashNode {
    int key; // Kernel identifier used as the lookup key
    KernelInfo* data;
    struct HashNode* next;
} HashNode;

// Hash-table structure
typedef struct {
    int size;
    HashNode** table; // Array of bucket pointers
} HashTable;

typedef struct {
    int smcount;
    int exp_sm;
    int kernel_nums;
    int ks_len;
    char outbuffer[256];
} KS;


extern HashTable* create_hash_table(int size);
extern unsigned int hash(int key, int size);
extern void ht_insert(HashTable* ht, int key, KernelInfo* data);
extern KernelInfo* ht_lookup(HashTable* ht, int key);
extern void free_hash_table(HashTable* ht);
// extern int load_kernel_db(HashTable* kernel_db, int smcount, const char* filename);
extern int load_kernel_db(HashTable* kernel_db, int smcount, int bs, const char* filename);
extern KernelInfo* find_ks_info(int id_to_find, HashTable* kernel_db);

#endif // CINFO_H_
