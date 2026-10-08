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
 * Hook CUDA function calls.
 */

/*
 * Defining _GNU_SOURCE allows us to call dlvsym().
 *
 * More on _GNU_SOURCE: https://stackoverflow.com/a/5583764
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* _GNU_SOURCE */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <inttypes.h>
#include <math.h>

#include "comm.h"
#include "common.h"
#include "cuda_defs.h"
#include "client.h"
#include "utlist.h"
// #include "cinfo.h"

// #include "saveinf.h"
// #include "predictor.h"

#define ENV_STARS_ENABLE_SINGLE_OVERSUB  "STARS_ENABLE_SINGLE_OVERSUB"

#define MEMINFO_RESERVE_MIB 1536           /* MiB */
#define KERN_SYNC_DURATION_BIG 1          /* seconds */
#define KERN_SYNC_WINDOW_STEPDOWN_THRESH 5 /* seconds */
#define KERN_SYNC_WINDOW_MAX 2048          /* Pending Kernels */
#define GPU_MAX_NUM 8					   /* Max GPU Numbers*/
#define MILLISEC (1000UL * 1000UL)

// Define the constant
#define SCHD_OVERHEAD 2.0  // Skip scheduling when the idle window is shorter than 2 ms
#define SYNCP_MESSAGE // Enable synchronization messages
#define EPSILON 1e-10


static void *real_dlsym_225(void *handle, const char *symbol);

cuCtxSynchronize_func real_cuCtxSynchronize = NULL;
cuLaunchKernel_func real_cuLaunchKernel = NULL;
cuMemcpy_func real_cuMemcpy = NULL;
cuMemcpyAsync_func real_cuMemcpyAsync = NULL;
cuMemcpyDtoH_func real_cuMemcpyDtoH = NULL;
cuMemcpyDtoHAsync_func real_cuMemcpyDtoHAsync = NULL;
cuMemcpyHtoD_func real_cuMemcpyHtoD = NULL;
cuMemcpyHtoDAsync_func real_cuMemcpyHtoDAsync = NULL;
cuMemcpyDtoD_func real_cuMemcpyDtoD = NULL;
cuMemcpyDtoDAsync_func real_cuMemcpyDtoDAsync = NULL;
cuGetProcAddress_func real_cuGetProcAddress = NULL;
cuGetProcAddress_v2_func real_cuGetProcAddress_v2 = NULL;
cuMemAllocManaged_func real_cuMemAllocManaged = NULL;
cuMemFree_func real_cuMemFree = NULL;
cuMemGetInfo_func real_cuMemGetInfo = NULL;
cuGetErrorString_func real_cuGetErrorString = NULL;
cuGetErrorName_func real_cuGetErrorName = NULL;
cuCtxSetCurrent_func real_cuCtxSetCurrent = NULL;
cuCtxGetCurrent_func real_cuCtxGetCurrent = NULL;

// Used by rate-measurement tests
cuCtxGetDevice_func real_cuCtxGetDevice = NULL;
cuDeviceGetUuid_func real_cuDeviceGetUuid = NULL;

cuEventCreate_func real_cuEventCreate = NULL;
cuEventRecord_func real_cuEventRecord = NULL;
cuEventQuery_func real_cuEventQuery = NULL;
cuEventSynchronize_func real_cuEventSynchronize = NULL;

// MPS-specific interception support
cuCtxCreate_v3_func real_cuCtxCreate_v3 = NULL;

cuInit_func real_cuInit = NULL;

nvmlDeviceGetUtilizationRates_func real_nvmlDeviceGetUtilizationRates = NULL;
nvmlInit_func real_nvmlInit = NULL;
nvmlDeviceGetHandleByIndex_func real_nvmlDeviceGetHandleByIndex = NULL;

size_t stars_size_mem_allocatable = 0;
size_t sum_allocated = 0;

int kern_since_sync = 0;
int all_kernel_num = 0;
int pending_kernel_window = 1;
pthread_mutex_t kcount_mutex;

int enable_single_oversub = 0;
int nvml_ok = 1;
int start_sysc_kernel = 0;


// STARS runtime behavior when MPS is enabled.
// Track the number of records written.
int records_written = 0;
int nowksid_sm = 0;
long long ksid_sm = 0;
char all_ksid_sm_str_out[256] = {0};
char outpath[256];
int kernel_nums = 1;
int ksid_count = 2;
int current_sm = 26;
int no_sm = 0;



// Global task data-size and SLO values
double submit_time = 0;
double task_datasize = 0;
double task_modeling_datasize= 0;  // Transfer volume used for modeling
double task_modeling_block = 0;
double task_modeling_thread = 0;
int task_slo = 0;
int task_modeling_slo= 0;  // SLO flag used for modeling
// Global continue-lock state. Receiving an SLO indicates that the client is initialized; subsequent requests use continue_with_lock_v2.
int continue_lock = 0;

struct timespec task_start_time = {0, 0};

/* Representation of a CUDA memory allocation */
struct cuda_mem_allocation {
	CUdeviceptr ptr;
	size_t size;
	struct cuda_mem_allocation *next;
};

/* Linked list that holds all memory allocations of current application. */
struct cuda_mem_allocation *cuda_allocation_list = NULL;

/* Initializaters will be executed only once per client application */
static pthread_once_t init_libstars_done = PTHREAD_ONCE_INIT;
static pthread_once_t init_done = PTHREAD_ONCE_INIT;

// Store the current rate
static int g_active_gpu[GPU_MAX_NUM] = {};
// Store the UUID of the active GPU
static CUuuid g_uuid[GPU_MAX_NUM];
static int g_gpu_id[GPU_MAX_NUM];


// Declarations
static void initialization(CUdevice);





// MPS-related state
// Manage at most four contexts
#define MAX_CONTEXTS 10
// Context metadata
typedef struct {
    CUcontext ctx;
    int smCount;
} ContextInfo;

// Global context pool
ContextInfo contextPool[MAX_CONTEXTS];

// Initialize the context pool during program startup
void init_context_pool() {
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        contextPool[i].ctx = NULL;
        contextPool[i].smCount = 0;
    }
}

// Mutex protecting access to contextPool
pthread_mutex_t pool_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * Handle get current device id
 */
int get_current_device_id() {
	CUdevice device = 0;
  	// Query the current device
	const CUresult ret = real_cuCtxGetDevice(&device);
	cuda_driver_check_error(ret, CUDA_SYMBOL_STRING(cuCtxGetDevice));
  	return device;
}


double get_current_now_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void init_csv(const char* outpath){
	FILE *fp = fopen(outpath, "a"); // "w" -> Write Text
    if (!fp) {
        perror("Failed to open file for text writing");
    }

    // Write the CSV header
    fprintf(fp, "allksid_sm, now_ksid_sm, duration\n");
	fclose(fp);
}


/**
 * @brief Save active_task_list to a CSV file.
 * @param outpath Destination file name.
 * @return Number of records written, or -1 on failure.
 */
void save_active_list_csv(const char* outpath, long long all_ksid_sm, int now_ksid_sm, int duration, char* ksid_sm_str) {
	FILE *fp = fopen(outpath, "a"); // "w" -> Write Text
    if (!fp) {
        perror("Failed to open file for text writing");
    }
	int result = fprintf(fp, "%s, %d, %d\n",
		ksid_sm_str,
		now_ksid_sm,
		duration);

	if (result < 0) {
		fprintf(stderr, "Error writing record %d to CSV file.\n", records_written);
		fclose(fp);
	}
	records_written++;

    fclose(fp);
    printf("Successfully wrote %d records to %s (CSV).\n", records_written, outpath);
}



// Initializer
static inline void initialization(CUdevice device) {
  g_active_gpu[device] = 1;

  fprintf(stderr, "initialize device %d\n", device);
  CUresult ret = real_cuDeviceGetUuid(&g_uuid[device], device);
  cuda_driver_check_error(ret, CUDA_SYMBOL_STRING(cuDeviceGetUuid));

  int gpu_id = 0;
  for (int i = 0; i < 16; ++i) {
    gpu_id += (int)g_uuid[device].bytes[i];
  }
  gpu_id = (gpu_id % 8 + 8) % 8;
  g_gpu_id[device] = gpu_id;

}

/* Load real CUDA {Driver API, NVML} functions and bootstrap auxiliary stuff. */
static void bootstrap_cuda(void)
{
	char *error;
	void *cuda_handle;
	void *nvml_handle;

	init_log_file();

	true_or_exit(pthread_mutex_init(&kcount_mutex, NULL) == 0);

	nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
	if (!nvml_handle) {
		error = dlerror();
		log_debug("%s", error);
		nvml_ok = 0;
	} else {
		dlerror();
		real_nvmlDeviceGetUtilizationRates =
			(nvmlDeviceGetUtilizationRates_func)real_dlsym_225(nvml_handle,
			CUDA_SYMBOL_STRING(nvmlDeviceGetUtilizationRates));
		error = dlerror();
		if (error != NULL) {
			log_debug("%s", error);
			nvml_ok = 0;
		}
		real_nvmlInit = (nvmlInit_func)
		real_dlsym_225(nvml_handle,CUDA_SYMBOL_STRING(nvmlInit));
		error = dlerror();
		if (error != NULL) {
			log_debug("%s", error);
			nvml_ok = 0;
		}
		real_nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_func)
		real_dlsym_225(nvml_handle,
			CUDA_SYMBOL_STRING(nvmlDeviceGetHandleByIndex));
		error = dlerror();
		if (error != NULL) {
			log_debug("%s", error);
			nvml_ok = 0;
		}
	}
	if (nvml_ok) log_debug("Found NVML");
	else log_debug("Could not find NVML");

	cuda_handle = dlopen("libcuda.so", RTLD_LAZY);
	if (!cuda_handle) {
		error = dlerror();
		log_fatal("%s", error);
	}
	/*
	 * For dlsym(), a return value of NULL does not necessarily indicate
	 * an error. Therefore, we must:
	 *  1. clear the previous error state by calling dlerror()
	 *  2. call dlsym()
	 *  3. call dlerror()
	 * If the value which dlerror() returns is not NULL, an error occured.
	 */
	dlerror();
	real_cuMemAllocManaged = (cuMemAllocManaged_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemAllocManaged));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemFree = (cuMemFree_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemFree));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuGetProcAddress = (cuGetProcAddress_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuGetProcAddress));
	error = dlerror();
	if (error != NULL)
		/*
		 * Print a debug message instead of failing immediately, since
		 * this symbol may not be used. This may be the case for CUDA
		 * Runtime <11.3.
		 */
		log_debug("%s", error);
	real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_func)
		real_dlsym_225(cuda_handle, CUDA_SYMBOL_STRING(cuGetProcAddress_v2));
	error = dlerror();
	if (error != NULL)
		/*
		 * Print a debug message instead of failing immediately, since
		 * this symbol may not be used. This may be the case for CUDA
		 * Runtime <12.0.
		 */
		log_debug("%s", error);

	/*event testing*/
	real_cuEventCreate = (cuEventCreate_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuEventCreate));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuEventRecord = (cuEventRecord_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuEventRecord));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuEventQuery = (cuEventQuery_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuEventQuery));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuEventSynchronize = (cuEventSynchronize_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuEventSynchronize));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);

	/*rate testing*/
	real_cuCtxGetDevice = (cuCtxGetDevice_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuCtxGetDevice));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuDeviceGetUuid = (cuDeviceGetUuid_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuDeviceGetUuid));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemGetInfo = (cuMemGetInfo_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemGetInfo));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuGetErrorString = (cuGetErrorString_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuGetErrorString));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuGetErrorName = (cuGetErrorString_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuGetErrorName));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuCtxSetCurrent = (cuCtxSetCurrent_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuCtxSetCurrent));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuCtxGetCurrent = (cuCtxGetCurrent_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuCtxGetCurrent));
	error = dlerror();
		if (error != NULL)
		log_fatal("%s", error);
	real_cuInit = (cuInit_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuInit));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuCtxSynchronize = (cuCtxSynchronize_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuCtxSynchronize));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuLaunchKernel = (cuLaunchKernel_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuLaunchKernel));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemcpy = (cuMemcpy_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemcpy));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemcpyAsync = (cuMemcpyAsync_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemcpyAsync));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemcpyDtoH = (cuMemcpyDtoH_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemcpyDtoH));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemcpyDtoHAsync = (cuMemcpyDtoHAsync_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemcpyDtoHAsync));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemcpyHtoD = (cuMemcpyHtoD_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemcpyHtoD));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemcpyHtoDAsync = (cuMemcpyHtoDAsync_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemcpyHtoDAsync));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemcpyDtoD = (cuMemcpyDtoD_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemcpyDtoD));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
	real_cuMemcpyDtoDAsync = (cuMemcpyDtoDAsync_func)
		real_dlsym_225(cuda_handle,CUDA_SYMBOL_STRING(cuMemcpyDtoDAsync));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);

	/* MPS-specific interception support */
	real_cuCtxCreate_v3 = (cuCtxCreate_v3_func)
	real_dlsym_225(cuda_handle, CUDA_SYMBOL_STRING(cuCtxCreate_v3));
	error = dlerror();
	if (error != NULL)
		log_fatal("%s", error);
}


/* Append a new CUDA memory allocation at the end of the list. */
static void insert_cuda_allocation(CUdeviceptr dptr, size_t bytesize)
{
	struct cuda_mem_allocation *allocation;


	sum_allocated += bytesize;
	// log_debug("Total allocated memory on GPU is %.2f MiB",
	// 	  toMiB(sum_allocated));

	true_or_exit(allocation = malloc(sizeof(*allocation)));

	allocation->ptr = dptr;
	allocation->size = bytesize;
	allocation->next = NULL;
	LL_APPEND(cuda_allocation_list, allocation);
}

/* Remove a CUDA memory allocation given the pointer it starts at */
static void remove_cuda_allocation(CUdeviceptr rm_ptr)
{
	struct cuda_mem_allocation *tmp, *a;


	LL_FOREACH_SAFE(cuda_allocation_list, a, tmp) {
		if (a->ptr == rm_ptr) {
			sum_allocated -= a->size;
			// log_debug("Total allocated memory on GPU is %.2f MiB",
			// 	  toMiB(sum_allocated));
			LL_DELETE(cuda_allocation_list, a);
			free(a);
		}
	}
}


/* Toggle debug mode and single process oversubscription based on envvars */
static void initialize_libstars(void)
{
	char *value;
	value = getenv(ENV_STARS_DEBUG);
	if (value != NULL)
		__debug = 1;
	value = getenv(ENV_STARS_ENABLE_SINGLE_OVERSUB);
	if (value != NULL) {
		enable_single_oversub = 1;
		log_warn("Enabling GPU memory oversubscription for this"
		         " application");
	}

	char *out_dir;
	out_dir = getenv(ENV_OUT_DIR);
	log_debug("[Hook]----ENV_OUT_DIR=%s", out_dir);

	snprintf(outpath, sizeof(outpath), "%sksid_sm_dur_summary_%d.csv", out_dir, 6);

	init_csv(outpath);

	bootstrap_cuda();
}


/*
 * Check the return value of a CUDA Driver API function call for errors.
 *
 * Interpret using the Driver API functions:
 * - cuGetErrorString
 * - cuGetErrorName
 */
void cuda_driver_check_error(CUresult err, const char *func_name)
{
	if (err != CUDA_SUCCESS) {
		const char *err_string;
		const char *err_name;
		real_cuGetErrorString(err, &err_string);
		real_cuGetErrorName(err, &err_name);
		log_warn("%s returned %s: %s",
	                 func_name, err_name, err_string);
	}
}


/*
 * Since we're interposing dlsym() in libstars, we use dlvsym() to obtain the
 * address of the real dlsym function.
 *
 * Depending on glibc version, we look for the appropriate symbol.
 *
 * Some context on the implementation:
 *
 * glibc 2.34 remove the internal __libc_dlsym() symbol that NVIDIA uses in
 * their cuHook example:
 * https://github.com/phrb/intro-cuda/blob/d38323b81cd799dc09179e2ef27aa8f81b6dac40/src/cuda-samples/7_CUDALibraries/cuHook/libcuhook.cpp#L43
 *
 * One solution, discussed in apitrace's repo is to use dlvsym(), which also
 * takes a version string as a 3rd argument, in order to obtain the real
 * dlsym().
 *
 * This is what user 'manisandro' suggested 8 years ago, when warning about
 * using the private __libc_dlsym():
 * https://github.com/apitrace/apitrace/issues/258
 *
 * The maintainer of the repo didn't heed the warning back then, it came back
 * 8 years later and bit them.
 *
 * This is also what user "derhass" suggests:
 * https://stackoverflow.com/a/18825060
 * (See section "UPDATE FOR 2021/glibc-2.34").
 *
 * Given all the above, we obtain the real `dlsym()` as such:
 * real_dlsym=dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
 *
 * Since we have to explicitly use a version argument in dlvsym(), we also have
 * to define and export two versions of dlsym (hence the linker script.), one
 * for each distinct glibc symbol version.
 *
 */
static void *real_dlsym_225(void *handle, const char *symbol)
{
	typedef void *(dlsym_t)(void *, const char *);
	static dlsym_t *r_dlsym;
	char *err;


	if (!r_dlsym) {
		dlerror();
		r_dlsym = (dlsym_t*)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
		err = dlerror();
		if (err != NULL)
			log_fatal("%s", err);
	}

	return (*r_dlsym)(handle, symbol);
}

static void *real_dlsym_234(void *handle, const char *symbol)
{
	typedef void *(dlsym_t)(void *, const char *);
	static dlsym_t *r_dlsym;
	char *err;


	if (!r_dlsym) {
		dlerror();
		r_dlsym = (dlsym_t*)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
		err = dlerror();
		if (err != NULL)
			log_fatal("%s", err);
	}

	return (*r_dlsym)(handle, symbol);
}


/*
 * CUDA Runtime API uses dlopen()/dlsym() to obtain addresses of the Driver API
 * functions.
 *
 * [spoiler: from CUDA 11.3 onwards, it only uses dlsym() to get the address
 *  of cuGetProcAddress() and then uses the latter to obtain the addresses
 *  of all other Driver API functions/symbols.]
 *
 * When the user program calls dlsym() requesting a Driver API symbol, return
 * our interposed version.
 *
 * In all other cases, call the real dlsym() from glibc and pass on the
 * requested symbol string.
 */
void *dlsym_225(void *handle, const char *symbol)
{
	if (strncmp(symbol, "cu", 2) != 0) {
		return (real_dlsym_225(handle, symbol));
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0) {
		return (void *)(&cuMemAlloc);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemFree)) == 0) {
		return (void *)(&cuMemFree);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemGetInfo)) == 0) {
		return (void *)(&cuMemGetInfo);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuGetProcAddress)) == 0) {
		return (void *)(&cuGetProcAddress);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuGetProcAddress_v2)) == 0) {
		return (void *)(&cuGetProcAddress_v2);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuInit)) == 0) {
		return (void *)(&cuInit);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuLaunchKernel)) == 0) {
		return (void *)(&cuLaunchKernel);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpy)) == 0) {
		return (void *)(&cuMemcpy);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyAsync)) == 0) {
		return (void *)(&cuMemcpyAsync);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyDtoH)) == 0) {
		return (void *)(&cuMemcpyDtoH);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyDtoHAsync)) == 0) {
		return (void *)(&cuMemcpyDtoHAsync);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyHtoD)) == 0) {
		return (void *)(&cuMemcpyHtoD);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyHtoDAsync)) == 0) {
		return (void *)(&cuMemcpyHtoDAsync);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyDtoD)) == 0) {
		return (void *)(&cuMemcpyDtoD);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyDtoDAsync)) == 0) {
		return (void *)(&cuMemcpyDtoDAsync);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuEventCreate)) == 0) {
		return (void *)(&cuEventCreate);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuEventRecord)) == 0) {
		return (void *)(&cuEventRecord);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuEventQuery)) == 0) {
		return (void *)(&cuEventQuery);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuEventSynchronize)) == 0) {
		return (void *)(&cuEventSynchronize);
	}
	else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxCreate_v3)) == 0) {
		return (void *)(&cuCtxCreate_v3);}


	return (real_dlsym_225(handle, symbol));
}

void *dlsym_234(void *handle, const char *symbol)
{
	if (strncmp(symbol, "cu", 2) != 0) {
		return (real_dlsym_234(handle, symbol));
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0) {
		return (void *)(&cuMemAlloc);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemFree)) == 0) {
		return (void *)(&cuMemFree);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemGetInfo)) == 0) {
		return (void *)(&cuMemGetInfo);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuGetProcAddress)) == 0) {
		return (void *)(&cuGetProcAddress);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuGetProcAddress_v2)) == 0) {
		return (void *)(&cuGetProcAddress_v2);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuInit)) == 0) {
		return (void *)(&cuInit);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuLaunchKernel)) == 0) {
		return (void *)(&cuLaunchKernel);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpy)) == 0) {
		return (void *)(&cuMemcpy);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyAsync)) == 0) {
		return (void *)(&cuMemcpyAsync);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyDtoH)) == 0) {
		return (void *)(&cuMemcpyDtoH);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyDtoHAsync)) == 0) {
		return (void *)(&cuMemcpyDtoHAsync);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyHtoD)) == 0) {
		return (void *)(&cuMemcpyHtoD);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyHtoDAsync)) == 0) {
		return (void *)(&cuMemcpyHtoDAsync);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyDtoD)) == 0) {
		return (void *)(&cuMemcpyDtoD);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemcpyDtoDAsync)) == 0) {
		return (void *)(&cuMemcpyDtoDAsync);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuEventCreate)) == 0) {
		return (void *)(&cuEventCreate);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuEventRecord)) == 0) {
		return (void *)(&cuEventRecord);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuEventQuery)) == 0) {
		return (void *)(&cuEventQuery);
	} else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuEventSynchronize)) == 0) {
		return (void *)(&cuEventSynchronize);
	}
	else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxCreate_v3)) == 0) {
		return (void *)(&cuCtxCreate_v3);
	}

	return (real_dlsym_234(handle, symbol));
}


/*
 * Older CUDA Runtime API (version <=11.2) does the following during internal
 * initialization (when the user program calls it for the first time):
 * 1. Calls dlopen("libcuda.so.1")
 * 2. Calls dlsym() for each function in the Driver API
 *
 * Newer CUDA Runtime API (version >=11.3) works like this:
 * 1. Calls dlopen("libcuda.so.1") and then dlsym("cuGetProcAddress")
 * 2. Calls cuGetProcAddress("cuGetProcAddress")
 *     1. If the pointer to "cuGetProcAddress" is NULL, it falls back to using
 *        dlsym() to get the Driver API function pointers
 *     2. If the pointer to "cuGetProcAddress" is not NULL, it uses
 *        cuGetProcAddress to get the Driver API function pointers.
 *
 * Interpose both, to cover all cases.
 *
 * The logic is the same as when interposing dlsym().
 */
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
	cuuint64_t flags)
{
	/*
	* cuGetProcAddress() will be called before cuInit() in CUDA
	* Runtime API (version >=11.3), so cuGetProcAddress() should also
	* serve as an entrypoint.
	* Otherwise, real_cuGetProcAddress may be a NULL pointer
	* when it is called.
	*/
	true_or_exit(pthread_once(&init_libstars_done, initialize_libstars) == 0);
	true_or_exit(pthread_once(&init_done, initialize_client) == 0);
	CUresult result = CUDA_SUCCESS;

	if (real_cuGetProcAddress == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	if (strcmp(symbol, "cuMemAlloc") == 0) {
		*pfn = (void *)(&cuMemAlloc);
	} else if (strcmp(symbol, "cuMemFree") == 0) {
		*pfn = (void *)(&cuMemFree);
	} else if (strcmp(symbol, "cuMemGetInfo") == 0) {
		*pfn = (void *)(&cuMemGetInfo);
	} else if (strcmp(symbol, "cuGetProcAddress") == 0) {
		*pfn = (void *)(&cuGetProcAddress);
	} else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
		*pfn = (void *)(&cuGetProcAddress_v2);
	} else if (strcmp(symbol, "cuInit") == 0) {
		*pfn = (void *)(&cuInit);
	} else if (strcmp(symbol, "cuLaunchKernel") == 0) {
		*pfn = (void *)(&cuLaunchKernel);
	} else if (strcmp(symbol, "cuMemcpy") == 0) {
		*pfn = (void *)(&cuMemcpy);
	} else if (strcmp(symbol, "cuMemcpyAsync") == 0) {
		*pfn = (void *)(&cuMemcpyAsync);
	} else if (strcmp(symbol, "cuMemcpyDtoH") == 0) {
		*pfn = (void *)(&cuMemcpyDtoH);
	} else if (strcmp(symbol, "cuMemcpyDtoHAsync") == 0) {
		*pfn = (void *)(&cuMemcpyDtoHAsync);
	} else if (strcmp(symbol, "cuMemcpyHtoD") == 0) {
		*pfn = (void *)(&cuMemcpyHtoD);
	} else if (strcmp(symbol, "cuMemcpyHtoDAsync") == 0) {
		*pfn = (void *)(&cuMemcpyHtoDAsync);
	} else if (strcmp(symbol, "cuMemcpyDtoD") == 0) {
		*pfn = (void *)(&cuMemcpyDtoD);
	} else if (strcmp(symbol, "cuMemcpyDtoDAsync") == 0) {
		*pfn = (void *)(&cuMemcpyDtoDAsync);
	} else if (strcmp(symbol, "cuEventCreate") == 0) {
		*pfn =  (void *)(&cuEventCreate);
	} else if (strcmp(symbol, "cuEventRecord") == 0) {
		*pfn =  (void *)(&cuEventRecord);
	} else if (strcmp(symbol, "cuEventQuery") == 0) {
		*pfn =  (void *)(&cuEventQuery);
	} else if (strcmp(symbol, "cuEventSynchronize") == 0) {
		*pfn =  (void *)(&cuEventSynchronize);
	}
	else if (strcmp(symbol, "cuCtxCreate_v3") == 0) {
		*pfn = (void *)(&cuCtxCreate_v3);
	}

	else {
		result = real_cuGetProcAddress(symbol, pfn, cudaVersion, flags);
	}

	return result;
}


CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
	cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)
{
	/*
	* cuGetProcAddress_v2() will be called before cuInit() in CUDA
	* Runtime API (version >=12.0), so cuGetProcAddress_v2()
	* should also serve as an entrypoint.
	*
	* Otherwise, real_cuGetProcAddress_v2 may be a
	* NULL pointer when it is called.
	*/
	true_or_exit(pthread_once(&init_libstars_done, initialize_libstars) == 0);
	true_or_exit(pthread_once(&init_done, initialize_client) == 0);
	CUresult result = CUDA_SUCCESS;

	if (real_cuGetProcAddress_v2 == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	/* This covers our custom "if" conditions.
	 * If we end up calling the real cuGetProcAddress_v2,
	 * it will overwrite this value.
	 */
	if (symbolStatus != NULL)
		*symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;

	if (strcmp(symbol, "cuMemAlloc") == 0) {
		*pfn = (void *)(&cuMemAlloc);
	} else if (strcmp(symbol, "cuMemFree") == 0) {
		*pfn = (void *)(&cuMemFree);
	} else if (strcmp(symbol, "cuMemGetInfo") == 0) {
		*pfn = (void *)(&cuMemGetInfo);
	} else if (strcmp(symbol, "cuGetProcAddress") == 0) {
		*pfn = (void *)(&cuGetProcAddress);
	} else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
		*pfn = (void *)(&cuGetProcAddress_v2);
	} else if (strcmp(symbol, "cuInit") == 0) {
		*pfn = (void *)(&cuInit);
	} else if (strcmp(symbol, "cuLaunchKernel") == 0) {
		*pfn = (void *)(&cuLaunchKernel);
	} else if (strcmp(symbol, "cuMemcpy") == 0) {
		*pfn = (void *)(&cuMemcpy);
	} else if (strcmp(symbol, "cuMemcpyAsync") == 0) {
		*pfn = (void *)(&cuMemcpyAsync);
	} else if (strcmp(symbol, "cuMemcpyDtoH") == 0) {
		*pfn = (void *)(&cuMemcpyDtoH);
	} else if (strcmp(symbol, "cuMemcpyDtoHAsync") == 0) {
		*pfn = (void *)(&cuMemcpyDtoHAsync);
	} else if (strcmp(symbol, "cuMemcpyHtoD") == 0) {
		*pfn = (void *)(&cuMemcpyHtoD);
	} else if (strcmp(symbol, "cuMemcpyHtoDAsync") == 0) {
		*pfn = (void *)(&cuMemcpyHtoDAsync);
	} else if (strcmp(symbol, "cuMemcpyDtoD") == 0) {
		*pfn = (void *)(&cuMemcpyDtoD);
	} else if (strcmp(symbol, "cuMemcpyDtoDAsync") == 0) {
		*pfn = (void *)(&cuMemcpyDtoDAsync);
	} else if (strcmp(symbol, "cuEventCreate") == 0) {
		*pfn =  (void *)(&cuEventCreate);
	} else if (strcmp(symbol, "cuEventRecord") == 0) {
		*pfn =  (void *)(&cuEventRecord);
	} else if (strcmp(symbol, "cuEventQuery") == 0) {
		*pfn =  (void *)(&cuEventQuery);
	} else if (strcmp(symbol, "cuEventSynchronize") == 0) {
		*pfn =  (void *)(&cuEventSynchronize);
	}
	else if (strcmp(symbol, "cuCtxCreate_v3") == 0) {
		*pfn = (void *)(&cuCtxCreate_v3);
	}
	else {
		result = real_cuGetProcAddress_v2(symbol, pfn, cudaVersion,
				                  flags, symbolStatus);
	}

	return result;
}



/*---------------- MPS-specific interception APIs ----------------*/

// Find a context by its SM count
CUcontext find_context_by_sm_count(int required_sm_count) {
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (contextPool[i].ctx != NULL && contextPool[i].smCount >= required_sm_count) {
            log_debug("Found context %p with %d SMs (required: %d)",
                   contextPool[i].ctx, contextPool[i].smCount, required_sm_count);
            return contextPool[i].ctx;
        }
    }

    log_warn("No context found with at least %d SMs", required_sm_count);
    return NULL;
}

void switch_context(int smcount) {

	// Synchronize before continuing
	CUresult cu_err = CUDA_SUCCESS;
	cu_err = real_cuCtxSynchronize();
	cuda_driver_check_error(cu_err, CUDA_SYMBOL_STRING(cuCtxSynchronize));
	log_debug("Synchronized current context before switching");

	// Select an appropriate context for smcount
	CUcontext current_ctx = find_context_by_sm_count(smcount);

	cu_err = real_cuCtxSetCurrent(current_ctx);
	cuda_driver_check_error(cu_err, CUDA_SYMBOL_STRING(cuCtxSetCurrent));
	log_debug("Switched to context with SM-%d", smcount);

	current_sm = smcount;
}


CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev)
{
	CUresult result = CUDA_SUCCESS;
	/* Return immediately if not initialized */
	if (real_cuCtxCreate_v3 == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	result = real_cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuCtxCreate_v3));


	// Add the new context to the pool
    CUcontext new_ctx = *pctx;
    log_debug("Successfully created new context with handle: %p", new_ctx);
	// Query the SM count
	int smCount = paramsArray[0].param.smCount.val;
	log_debug("New context SM count: %d", smCount);

    // Lock the mutex for thread-safe access
    pthread_mutex_lock(&pool_mutex);

    int already_exists = 0;
    int next_free_index = -1;

		// Scan for an existing entry and the first free slot
	for (int i = 0; i < MAX_CONTEXTS; i++) {
		// Check whether the context already exists
		if (contextPool[i].ctx == new_ctx) {
			already_exists = 1;
			break; // Found; stop searching
		}
		// Record the first free slot
		if (contextPool[i].ctx == NULL && next_free_index == -1) {
			next_free_index = i;
		}
	}

	if (already_exists) {
		// Emit a warning if the context already exists
		log_warn("Context %p already exists in the pool. Not adding it again.", new_ctx);
	} else {
		// Otherwise, add it to the first free slot
		if (next_free_index != -1) {
			contextPool[next_free_index].ctx = new_ctx;
			contextPool[next_free_index].smCount = smCount;
			log_info("New context %p with %d SMs added to pool at index %d.",
					new_ctx, smCount, next_free_index);
		} else {
			// No free slot means the pool is full
			log_warn("Context pool is full! Cannot add new context %p. This may lead to resource leaks.", new_ctx);
			// The context has been created but is not tracked by the pool.
		}
	}
    // Unlock the mutex
    pthread_mutex_unlock(&pool_mutex);

	create_SM_table(smCount, no_sm);
	no_sm ++;

	current_sm = smCount;
	return result;
}


//
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
{
	static int got_max_mem_size = 0;
	size_t junk;
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemAllocManaged == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	if (got_max_mem_size == 0) {
		result = cuMemGetInfo(&stars_size_mem_allocatable, &junk);
		cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemGetInfo));
		got_max_mem_size = 1;
	}

	if ((sum_allocated + bytesize) > stars_size_mem_allocatable) {
		if (enable_single_oversub == 0) {
			return CUDA_ERROR_OUT_OF_MEMORY;
		} else {
			log_warn("Memory allocations exceeded physical GPU"
				 " memory capacity. This can cause extreme"
				 " performance degradation!");
		}
	}

	// log_debug("cuMemAlloc requested %zu bytes", bytesize);
	result = real_cuMemAllocManaged(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemAllocManaged));
	// log_debug("cuMemAllocManaged allocated %zu bytes at 0x%llx",
	// 	bytesize, *dptr);
	if (result == CUDA_SUCCESS) {
		insert_cuda_allocation(*dptr, bytesize);
	}

	return result;
}


CUresult cuMemFree(CUdeviceptr dptr)
{
	CUresult result = CUDA_SUCCESS;


	if (real_cuMemFree == NULL) return CUDA_ERROR_NOT_INITIALIZED;
	result = real_cuMemFree(dptr);
	if (result == CUDA_SUCCESS) remove_cuda_allocation(dptr);

	return result;
}


CUresult cuMemGetInfo(size_t *free, size_t *total)
{
	long long reserve_mib;
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemGetInfo == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	result = real_cuMemGetInfo(free, total);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemGetInfo));

	log_debug("real_cuMemGetInfo returned free=%.2f MiB, total=%.2f MiB",
		 toMiB(*free), toMiB(*total));

        /*
	 * Hide a static amount of GPU memory from the applications. CUDA uses
	 * this memory to store context information and it is not pageable.
	 *
	 * In practice, this amount of memory is not static and depends on
	 * the number of colocated applications. Each one has its own context,
	 * which eats away some physical, non-pageable GPU memory.
	 *
	 * The first application that runs theoretically has (TOTAL_GPU_MEM -
	 * CONTEXT_SIZE) memory available.
	 *
	 * CONTEXT_SIZE typically uses a few hundred MB and depends on the GPU
	 * model.
	 *
	 * cuBLAS and other CUDA libraries also eat away at this memory.
	 *
	 * When another app runs, this "working memory size" shrinks further
	 * and can lead to thrashing within the first application, even when
	 * it runs alone.
	 *
	 * We cannot shrink the memory allocations of a running app, and the
	 * app thinks all of its memory is physically backed, since it's
	 * programmed with cuMemAlloc semantics in mind.
	 *
	 * To avoid internal thrashing, we empirically choose a sane value for
	 * MEMINFO_RESERVE_MIB.
	 */
	reserve_mib = (MEMINFO_RESERVE_MIB) MiB;
	*free = *total - (size_t) reserve_mib;

	log_debug("STARS cuMemGetInfo returning free=%.2f MiB,"
		  " total=%.2f MiB", toMiB(*free), toMiB(*total));
	return result;
}

/*
 * A call to cuInit is an indicator that the present application is a CUDA
 * application and that we should bootstrap STARS.
 */
CUresult cuInit(unsigned int flags)
{
	CUresult result = CUDA_SUCCESS;

	true_or_exit(pthread_once(&init_libstars_done, initialize_libstars) == 0);
	true_or_exit(pthread_once(&init_done, initialize_client) == 0);

	result = real_cuInit(flags);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuInit));
	no_sm = 0;

	return result;
}


CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
	unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX,
	unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void **kernelParams,
	void **extra)
{
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuLaunchKernel == NULL) return CUDA_ERROR_NOT_INITIALIZED;


	// log_debug("[HOOK]----task_slo %f \n", task_slo);
	if (!(continue_lock)){
		// log_debug("[HOOK]----task_slo %d \n", task_slo);
		if (fabs(task_slo) > EPSILON)
		{
			if(task_slo > 1){
				continue_lock = 1; // Model activated
			}

		}
	}
	// 	// printf("********--------------");
	// }else{
	// 	continue_with_lock_v2(0.0, 0.0, 0.0, 0.0);
	// 	// printf("*continue_lock  == %d", continue_lock);
	// }

	// continue_with_lock();

	// struct timespec req_time = {0, 0};
	// struct timespec rec_time = {0, 0};
    // struct timespec duration = {0, 0};
	// unsigned int elapsed_ms;
	// true_or_exit(clock_gettime(CLOCK_MONOTONIC, &req_time) == 0);

	result = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
		blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

	// true_or_exit(clock_gettime(CLOCK_MONOTONIC, &rec_time) == 0);
	// timespecsub(&rec_time, &req_time, &duration);
	// elapsed_ms = ((unsigned int)duration.tv_sec * 1000)
	// 					+ (duration.tv_nsec / 1000000);
	// log_debug("Kernel execution time: %.12d ms", elapsed_ms);

	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuLaunchKernel));

	// true_or_exit(pthread_mutex_lock(&kcount_mutex) == 0);


	// if (start_sysc_kernel == 1){
	// 	kern_since_sync++;
	// 	if (kern_since_sync >= pending_kernel_window) {
	// 		struct timespec cuda_cuda_sync_start_time = {0, 0};
	// 		struct timespec cuda_sync_complete_time = {0, 0};
	// 		struct timespec cuda_sync_duration = {0, 0};
	// 		true_or_exit(clock_gettime(CLOCK_MONOTONIC, &cuda_cuda_sync_start_time) == 0);
	// 		result = real_cuCtxSynchronize();
	// 		cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuCtxSynchronize));
	// 		true_or_exit(clock_gettime(CLOCK_MONOTONIC, &cuda_sync_complete_time) == 0);
	// 		timespecsub(&cuda_sync_complete_time, &cuda_cuda_sync_start_time, &cuda_sync_duration);

	// 		/*
	// 		* Possibly a series of huge kernels. We cannot risk to
	// 		* simply fall back to previous window. Fall back to
	// 		* the initial window of 1.
	// 		*/
	// 		if (cuda_sync_duration.tv_sec >= KERN_SYNC_DURATION_BIG)
	// 			pending_kernel_window = 1;

	// 		/*
	// 		* Intermediate situation, don't be too harsh. Rein the
	// 		* rate in.
	// 		*/
	// 		else if ((cuda_sync_duration.tv_sec * 1000 + cuda_sync_duration.tv_nsec / 1000000)  >= KERN_SYNC_WINDOW_STEPDOWN_THRESH)
	// 			pending_kernel_window = max(pending_kernel_window/2, 1);

	// 		/*
	// 		* Max window size is simply a heuristic.
	// 		*/
	// 		else pending_kernel_window = min(pending_kernel_window + 1,
	// 								KERN_SYNC_WINDOW_MAX);

	// 		log_debug("Pending Kernel Window is %d.", pending_kernel_window);
	// 		kern_since_sync = 0;
	// 	}
	// }


	// true_or_exit(pthread_mutex_unlock(&kcount_mutex) == 0);

	// static struct timespec ks_start_time = {0, 0};

	/* Send a request when an MPS kernel segment completes. */
	// log_debug("Current kernel count: %d/%d", kern_since_sync, kernel_nums);
	kern_since_sync++;
	all_kernel_num++;

	// if((kern_since_sync == (kernel_nums / 2)) && (continue_lock == 1)){
	// 	char *env_value1 = getenv(SH);
	// 	int sh_value1 = atoi(env_value1);
	// 	if (sh_value1 != 2){
	// 		switch_context(16);
	// 	}else{
	// 		switch_context(10);
	// 	}

	// 	// Four ResNet workloads
	// 	// switch_context(26);
	// }

	if((kern_since_sync == 25) && (continue_lock == 1)){
		switch_context(26);
	}


	if((kern_since_sync == 50) && (continue_lock == 1)){
		switch_context(current_sm);

		// Scheduling-independent timestamp recording
		// struct timespec cuda_cuda_sync_start_time = {0, 0};
		// struct timespec cuda_cuda_sync_new_start_time = {0, 0};
		// struct timespec cuda_sync_complete_time = {0, 0};
		// struct timespec cuda_sync_duration = {0, 0};
		// struct timespec ks_duration = {0, 0};
		// true_or_exit(clock_gettime(CLOCK_MONOTONIC, &cuda_cuda_sync_start_time) == 0);
		// result = real_cuCtxSynchronize();
		// cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuCtxSynchronize));

		// true_or_exit(clock_gettime(CLOCK_MONOTONIC, &cuda_sync_complete_time) == 0);
		// timespecsub(&cuda_sync_complete_time, &cuda_cuda_sync_start_time, &cuda_sync_duration);
		// // log_debug("%d-th KS [Sync] Duration--------%lldns", ksid_count, cuda_sync_duration.tv_nsec);


		// // Compute this segment's duration from consecutive completion times
        // timespecsub(&cuda_sync_complete_time, &ks_start_time, &ks_duration);

		// log_debug("%d-th KS Duration--------%lldns", ksid_count, ks_duration.tv_nsec);
		// Scheduling-independent timestamp recording




		// Scheduling-independent data capture for later analysis */
		// nowksid_sm = ksid_count*100 + current_sm;
		// save_active_list_csv(outpath, ksid_sm, nowksid_sm, ks_duration.tv_nsec, all_ksid_sm_str_out);
		// log_debug("Current kernel-segment/SM data: %d", ksid_sm);
		// Scheduling-independent data capture for later analysis */

		// KS* ks = mps_continue_with_lock(current_sm, ksid_count, task_slo);
		// KS* ks = mps_continue_with_lock(current_sm, ksid_count, task_slo, submit_time);
		// kernel_nums = ks->kernel_nums;
		// current_sm = ks->smcount;

		// Scheduling-independent data capture
		// memcpy(all_ksid_sm_str_out, ks->outbuffer, sizeof(all_ksid_sm_str_out));
		// Scheduling-independent data capture

		// TODO: Switch contexts

		// switch_context(ks->smcount);
		// kern_since_sync=0;
		// ksid_count += ks->ks_len;

		// Record completion time for the next segment-duration calculation
		// Scheduling-independent timestamp recording
		// true_or_exit(clock_gettime(CLOCK_MONOTONIC, &cuda_cuda_sync_new_start_time) == 0);
		// ks_start_time = cuda_cuda_sync_new_start_time;
		// Scheduling-independent timestamp recording
	}


	return result;
}


/*
 * Memory copy functions can affect the resident pages on GPU, so we must
 * block them as well when the client doesn't have the GPU lock.
 */
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
	CUresult result = CUDA_SUCCESS;


	if (real_cuMemcpy == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	// continue_with_lock();

	result = real_cuMemcpy(dst, src, ByteCount);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemcpy));

	log_debug("cuMemcpy copy=%.8f MiB", toMiB(ByteCount));

	return result;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
	CUstream hStream)
{
	CUresult result = CUDA_SUCCESS;


	if (real_cuMemcpyAsync == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	// continue_with_lock();

	result = real_cuMemcpyAsync(dst, src, ByteCount, hStream);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemcpyAsync));

	return result;
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemcpyDtoH == NULL) return CUDA_ERROR_NOT_INITIALIZED;
	result = real_cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemcpyDtoH));
	// host_sync_call();

	return result;
}

CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
	size_t ByteCount, CUstream hStream)
{
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemcpyDtoHAsync == NULL) return CUDA_ERROR_NOT_INITIALIZED;


	result = real_cuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemcpyDtoHAsync));
	// host_sync_call();


	// Reset the kernel-segment ID for MPS.
	ksid_count = 2;
	kernel_nums = 1;
	kern_since_sync = 0;
	all_kernel_num = 0;


	release_lock();

	return result;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost,
	size_t ByteCount)
{
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemcpyHtoD == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	// continue_with_lock();
	result = real_cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemcpyHtoD));
	// log_debug("cuMemcpyHtoD copy=%.8f MiB",toMiB(ByteCount));
	// // host_sync_call();

	return result;
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
	size_t ByteCount, CUstream hStream)
{
	// log_debug("berfore cuMemcpyHtoDAsync copy=%.8f MiB, Stream=%p", toMiB(ByteCount), hStream);
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemcpyHtoDAsync == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	// continue_with_lock();
	result = real_cuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemcpyHtoDAsync));
	// log_debug("after cuMemcpyHtoDAsync copy=%.8f KiB, Stream=%p", toKiB(ByteCount), hStream);
	if (toKiB(ByteCount) == 0.00390625){
		const int *data = (const int *)srcHost;
		int slo = *data;
		log_debug("after cuMemcpyHtoDAsync slo=%d", slo);
		task_slo = slo;
		log_debug("after cuMemcpyHtoDAsync slo=%d", slo);
		submit_time = get_current_now_time_ms();
		kernel_nums = 1;
		ksid_count = 2;
		kern_since_sync = 0;

		continue_lock = 0;
		if(task_slo == 1){
			true_or_exit(clock_gettime(CLOCK_REALTIME, &task_start_time) == 0);
		}
		log_debug("after cuMemcpyHtoDAsync task_slo=%f", task_slo);

	}

	if (toKiB(ByteCount) > 5)
	{
		// saveinf("cuMemcpyHtoDAsync copy=%.8f KiB", toKiB(ByteCount))
		// log_debug("after cuMemcpyHtoDAsync copy=%.8f KiB, Stream=%p", toKiB(ByteCount), hStream);
		// log_save("after cuMemcpyHtoDAsync copy=%.8f KiB, Stream=%p", toKiB(ByteCount), hStream);
		task_datasize = toKiB(ByteCount);
	}

	// host_sync_call();

	return result;
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
	size_t ByteCount)
{
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemcpyDtoD == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	// continue_with_lock();
	result = real_cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemcpyDtoD));

	return result;
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
	size_t ByteCount, CUstream hStream)
{
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemcpyDtoDAsync == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	// continue_with_lock();
	result = real_cuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemcpyDtoDAsync));

	return result;
}


CUresult cuEventRecord(CUevent hEvent, CUstream hStream){
    CUresult result = CUDA_SUCCESS;
	/* Return immediately if not initialized */
	if (real_cuEventRecord == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	result = real_cuEventRecord(hEvent, hStream);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuEventRecord));
	return result;
}

CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags){
    CUresult result = CUDA_SUCCESS;
	/* Return immediately if not initialized */
	if (real_cuEventCreate == NULL) return CUDA_ERROR_NOT_INITIALIZED;
	result = real_cuEventCreate(phEvent, Flags);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuEventCreate));

	return result;
}

CUresult cuEventQuery(CUevent hEvent){
    CUresult result = CUDA_SUCCESS;
	/* Return immediately if not initialized */
	if (real_cuEventQuery == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	result = real_cuEventQuery(hEvent);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuEventQuery));

	return result;
}

CUresult cuEventSynchronize(CUevent hEvent){
    CUresult result = CUDA_SUCCESS;
	/* Return immediately if not initialized */
	if (real_cuEventSynchronize == NULL) return CUDA_ERROR_NOT_INITIALIZED;
	result = real_cuEventSynchronize(hEvent);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuEventSynchronize));
	// end_event();
	start_sysc_kernel = 0;
	// pending_kernel_window = 1;
	continue_lock = 0;
	if(task_modeling_slo == 1){
		struct timespec task_complete_time = {0, 0};
		struct timespec task_duration = {0, 0};
		true_or_exit(clock_gettime(CLOCK_REALTIME, &task_complete_time) == 0);
		timespecsub(&task_complete_time, &task_start_time, &task_duration);
		double task_modeling_latency = (task_duration.tv_sec * 1000 + task_duration.tv_nsec / 1e6);
		// predict_modeling(task_modeling_datasize, task_modeling_block, task_modeling_thread, task_modeling_latency);

		task_modeling_block = 0;
		task_modeling_thread = 0;
		task_modeling_datasize = 0;
		task_modeling_slo = 0;
	}

	return result;
}





__asm__(".symver dlsym_225, dlsym@@GLIBC_2.2.5");
__asm__(".symver dlsym_234, dlsym@GLIBC_2.34");
