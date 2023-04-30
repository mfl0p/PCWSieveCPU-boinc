
// cpu_sieve.h

typedef struct {

	uint64_t pmin = 0, pmax = 0, checksum = 0, primecount = 0;
	uint64_t workunit, r0, r1, lastN;
	double wu_range;

//	tpsieve option -M2, change K's modulus to 2
	uint32_t kstep = 2, koffset = 1;
//	default for twin prime search is kstep = 6, koffset = 3

	uint32_t factorcount = 0, threads = 1, nmin = 0, nmax = 0, kmin = 0, kmax = 0;  // note k is 32 bit!
	uint32_t nstep, mont_nstep, num_results;
	int32_t bbits, bbits1;
	bool cw = false, test = false, avx512 = false;

}searchData;


// 1 if a number mod 15 is not divisible by 2 or 3.
//                      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
const int prime15[] = { 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1 };


/* shared data between threads */
//-------------------------------
extern uint64_t g_primecount;
extern uint64_t g_checksum;
extern pthread_mutex_t lock1;
//-------------------------------
extern uint64_t g_complete;
extern pthread_mutex_t lock2;
//-------------------------------


typedef struct _thread_data_t {
	uint64_t pmin,pmax;
	int id;
	searchData sd;
} thread_data_t;


extern int read_state_thread( searchData sd, uint64_t & current_p, int thread_id, bool & state_a, uint64_t & primecount, uint64_t & checksum, uint64_t & last_trickle );

extern void checkpoint_thread( searchData sd, uint64_t my_P, int thread_id, bool & state_a, uint64_t primecount, uint64_t checksum, uint64_t & last_trickle );

extern void clear_temp_factors_file( int thread_id );

extern void report_solution_temp( char * results, int thread_id );

extern void ckerr(int err);

extern bool goodfactor_cw(uint32_t uk, uint32_t n, int c);

extern bool goodfactor(uint32_t uk, uint32_t n, int c);

extern void *thr_func_cw_avx512(void *arg);

extern void *thr_func_avx512(void *arg);

extern void *thr_func_cw(void *arg);

extern void *thr_func(void *arg);

extern void cpu_sieve( searchData & sd );

extern void run_test( searchData & sd );
