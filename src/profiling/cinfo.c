#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cinfo.h"
#include "common.h"


// Create a hash table
HashTable* create_hash_table(int size) {
    HashTable* ht = malloc(sizeof(HashTable));

    if (ht == NULL) { // Check malloc
        log_warn("Failed to allocate memory for HashTable");
        return NULL;
    }

    ht->size = size;
    ht->table = calloc(size, sizeof(HashNode*));
    if (ht->table == NULL) { // Check calloc
        log_warn("Failed to allocate memory for HashTable table");
        free(ht); // Release the previously allocated table structure
        return NULL;
    }
    // log_info("[cinfo]----sucess create table");
    return ht;
}

// Simple hash function
unsigned int hash(int key, int size) {
    int index = (unsigned int)key % size;
    // if (index < 0) index += ht->size;
    return index;
}

// Insert data into the hash table
void ht_insert(HashTable* ht, int key, KernelInfo* data) {
    // Validate the supplied hash-table pointer before accessing it
    if (ht == NULL || ht->table == NULL) {
        log_warn("Error: Attempted to insert into a NULL or invalid hash table.\n");
        return;
    }

    unsigned int index = hash(key, ht->size);

    // Create a new node
    HashNode* newHashNode = malloc(sizeof(HashNode));
    if (newHashNode == NULL) { // Check malloc
        log_warn("Failed to allocate memory for new hash node");
        // TODO: Propagate allocation failure to the caller; avoid dereferencing NULL
        return;
    }

    // log_info("[cinfo]----------insert key=%d index=%d ht->size=%d\n", key, index, ht->size);

    HashNode* new_node = malloc(sizeof(HashNode));
    newHashNode->key = key;
    newHashNode->data = data;

    // Insert at the head of the linked list
    if (ht->table[index] == NULL) {
        // log_info("insert: head");
        ht->table[index] = newHashNode;
    } else {
        // log_info("insert: head not null");
        newHashNode->next = ht->table[index];
        ht->table[index] = newHashNode;
    }
}

// Look up data in the hash table
KernelInfo* ht_lookup(HashTable* ht, int key) {
    unsigned int index = hash(key, ht->size);
    HashNode* current = ht->table[index];
    // log_info("[cinfo]----------lookup key=%d index=%d ht->size=%d\n", key, index, ht->size);

    // Search the linked list
    while (current != NULL) {
        // log_info("[cinfo]----------current not null");
        if (current->key == key) {
            return current->data; // Found
        }
        current = current->next;
    }
    return NULL; // Not found
}

// Free the hash table
void free_hash_table(HashTable* ht) {
    for (int i = 0; i < ht->size; i++) {
        HashNode* current = ht->table[i];
        while (current != NULL) {
            HashNode* temp = current;
            current = current->next;
            free(temp->data); // Free the KernelInfo structure
            free(temp);     // Free the node
        }
    }
    free(ht->table);
    free(ht);
}


// Load CSV records into the hash table
int load_kernel_db(HashTable* kernel_db, int smcount, int bs, const char* filename) {
    // Open the CSV file for reading
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening file");
        return 0;
    }

    char line[MAX_LINE_LEN];
    // Skip the header
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }


    int lastline = 1;
    double drame = 0.0;
    double l2 = 0.0;
    int dur = 0;

    // Read and parse one line at a time
    while (fgets(line, sizeof(line), fp)) {
        // Allocate a new record
        KernelInfo* new_ks = calloc(1, sizeof(KernelInfo));
        if (!new_ks) {
            log_warn("Memory allocation failed\n");
            break;
        }


        // Parse with sscanf
        int items = sscanf(line, "%d,%d,%d,%lf,%lf,%lf,%d,%d",
               &new_ks->ks_id,
               &new_ks->kernel_num,
               &new_ks->duration_ns,
               &new_ks->sm_usage,
               &new_ks->dram_demand,
               &new_ks->l2_demand,
               &new_ks->now_duration_ns,
               &new_ks->solo_duration_ns
            );
        // Encode smcount and batch size in ks_id; each uses two decimal digits.
        new_ks->ks_id = new_ks->ks_id * 10000 + smcount * 100 + bs;

        drame = new_ks->dram_demand;
        l2 = new_ks->l2_demand;
        dur = new_ks->duration_ns;

        // log_info("[cinfo]----, ks_id: %d, kernel_num: %d, duration: %d, sm_usage: %.2f, dram: %.2f, l2: %.2f\n",
        //     new_ks->ks_id,
        //     new_ks->kernel_num,
        //     new_ks->duration_ns,
        //     new_ks->sm_usage,
        //     new_ks->dram_demand,
        //     new_ks->l2_demand);

        if (items == 8) {
            // Insert successfully parsed records into the hash table
            // log_info("[cinfo]----sucess parse line %d", lastline);
            ht_insert(kernel_db, new_ks->ks_id, new_ks);
        } else {
            // Release records that failed to parse
            log_warn("Parsing error on line %d: expected 8 items, got %d\n", lastline, items);
            free(new_ks);
        }
        lastline++;
    }

    // Append a sentinel record to identify the end of the kernel sequence
    KernelInfo* new_ks = malloc(sizeof(KernelInfo));
    // new_ks->ks_id =  smcount * 100 + lastline;
    new_ks->ks_id =  lastline * 10000 + smcount * 100 + bs;
    new_ks->kernel_num = 0;
    new_ks->duration_ns = dur;
    new_ks->sm_usage = 0;
    new_ks->dram_demand = drame;
    new_ks->l2_demand = l2;
    new_ks->now_duration_ns = 0;
    new_ks->solo_duration_ns = 0;

    ht_insert(kernel_db, new_ks->ks_id, new_ks);

    fclose(fp);

    return lastline;
}

KernelInfo* find_ks_info(int id_to_find, HashTable* kernel_db)
{
    // log_info("\nSearching for kernel with ID: %d\n", id_to_find);

    KernelInfo* found_kernel = ht_lookup(kernel_db, id_to_find);

    // if (found_kernel) {
    //     log_info("[cinfo]---------Found! Kernel Name: %d, Duration: %lld ns, Dram: %.2f, L2: %.2f\n",
    //            found_kernel->kernel_num, found_kernel->duration_ns, found_kernel->dram_demand, found_kernel->l2_demand);
    // } else {
    //     log_info("Kernel with ID %d not found.\n", id_to_find);
    // }
    return found_kernel;
}
