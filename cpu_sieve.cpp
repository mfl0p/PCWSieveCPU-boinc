/*
	PCWSieve
	Bryan Little, Feb 21 2023
	
	Search algorithm by
	Geoffrey Reynolds, 2009
	Ken Brazier, 2009
	https://github.com/Ken-g6/PSieve-CUDA/tree/redcl
	https://github.com/Ken-g6/PSieve-CUDA/tree/cw

	With contributions by
	Yves Gallot

*/

#include <unistd.h>
#include <pthread.h>

#include "boinc_api.h"

#include "primesieve.h"
#include "factor_proth.h"
#include "verify_factor.h"
#include "putil.h"
#include "cpu_sieve.h"

using namespace std;


/* shared data between threads */
//-------------------------------
uint64_t g_primecount;
uint64_t g_checksum;
pthread_mutex_t lock1;
//-------------------------------
uint64_t g_complete;
pthread_mutex_t lock2;
//-------------------------------


void handle_trickle_up(uint64_t & last_trickle)
{
	if(boinc_is_standalone()) return;

	uint64_t now = (uint64_t)time(NULL);

	if( (now-last_trickle) > 86400 ){	// Once per day

		last_trickle = now;

		double progress = boinc_get_fraction_done();
		double cpu;
		boinc_wu_cpu_time(cpu);
		APP_INIT_DATA init_data;
		boinc_get_init_data(init_data);
		double run = boinc_elapsed_time() + init_data.starting_elapsed_time;

		char msg[512];
		sprintf(msg, "<trickle_up>\n"
			    "   <progress>%lf</progress>\n"
			    "   <cputime>%lf</cputime>\n"
			    "   <runtime>%lf</runtime>\n"
			    "</trickle_up>\n",
			     progress, cpu, run  );
		char variety[64];
		sprintf(variety, "cwsieve_progress");
		boinc_send_trickle_up(variety, msg);
	}

}


// resolve is used for final factors.txt that is uploaded to BOINC server
FILE *boinc_res_fopen(const char * filename, const char * mode)
{
	char resolved_name[512];

	boinc_resolve_filename(filename,resolved_name,sizeof(resolved_name));

	return boinc_fopen(resolved_name,mode);

}


void gatherfactors( searchData & sd ){
	FILE *in;
	FILE *out;
	char filename[20];
	int len = 256;
	char input[len];

	out = boinc_res_fopen("factors.txt","w");

	if( out == NULL ){
		fprintf(stderr,"Cannot open factors.txt !!!\n");
		exit(EXIT_FAILURE);
	}

	// collect factors from each thread's file
	for(uint32_t i=0; i<sd.threads; ++i){
		if ( sprintf( filename, "f%d.tmp",i) < 0 ){
			fprintf(stderr,"error in sprintf()\n");
			exit(EXIT_FAILURE);
		}
		in = boinc_fopen(filename,"r");

		if( in == NULL ){
			fprintf(stderr,"Cannot open %s !!!\n",filename);
			exit(EXIT_FAILURE);
		}

		while( fgets(input, len, in) ){
			if( fprintf( out, "%s", input ) < 0 ){
				fprintf(stderr,"Cannot write to factors.txt !!!\n");
				exit(EXIT_FAILURE);
			}	
			++sd.factorcount;
		}
		fclose(in);
	}

	// print checksum to end of factors file
	if(sd.factorcount == 0){
		if( fprintf( out, "no factors\n%016" PRIX64 "\n", g_checksum ) < 0 ){
			fprintf(stderr,"error in fprintf()\n");
			exit(EXIT_FAILURE);
		}
	}
	else{
		if( fprintf( out, "%016" PRIX64 "\n", g_checksum ) < 0 ){
			fprintf(stderr,"error in fprintf()\n");
			exit(EXIT_FAILURE);
		}
	}

	fclose(out);


}


/* Return 1 only if a valid checkpoint can be read.
   Attempts to read from both state files,
   uses the most recent one available.
 */
int read_state_thread( searchData sd, uint64_t & current_p, int thread_id, bool & state_a, uint64_t & primecount, uint64_t & checksum, uint64_t & last_trickle)
{
	FILE *in;
	bool good_state_a = true;
	bool good_state_b = true;
	uint64_t workunit_a, workunit_b;
	uint64_t current_a, current_b;
	uint64_t primecount_a, primecount_b;
	uint64_t checksum_a, checksum_b;
	uint32_t threads_a, threads_b;
	uint64_t trickle_a, trickle_b;
	char filenameA[20];
	char filenameB[20];

	if ( sprintf( filenameA, "sA%d.tmp", thread_id) < 0 ){
		fprintf(stderr,"error in sprintf()\n");
		exit(EXIT_FAILURE);
	}
	if ( sprintf( filenameB, "sB%d.tmp", thread_id) < 0 ){
		fprintf(stderr,"error in sprintf()\n");
		exit(EXIT_FAILURE);
	}

        // Attempt to read state file A
	if ((in = boinc_fopen(filenameA,"r")) == NULL){
		good_state_a = false;
        }
	else if (fscanf(in,"%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %u %" PRIu64 "\n",&workunit_a,&current_a,&primecount_a,&checksum_a,&threads_a,&trickle_a) != 6){
		fprintf(stderr,"Cannot parse %s !!!\n",filenameA);
		good_state_a = false;
	}
	else{
		fclose(in);
		/* Check that start stop match and thread count of checkpoints equals current thread count */
		if (workunit_a != sd.workunit || threads_a != sd.threads){
			good_state_a = false;
		}
	}

        // Attempt to read state file B
        if ((in = boinc_fopen(filenameB,"r")) == NULL){
                good_state_b = false;
        }
	else if (fscanf(in,"%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %u %" PRIu64 "\n",&workunit_b,&current_b,&primecount_b,&checksum_b,&threads_b,&trickle_b) != 6){
                fprintf(stderr,"Cannot parse %s !!!\n",filenameB);
                good_state_b = false;
        }
        else{
                fclose(in);
		/* Check that start stop match and thread count of checkpoints equals current thread count */
		if (workunit_b != sd.workunit || threads_b != sd.threads){
                        good_state_b = false;
                }
        }

        // If both state files are OK, check which is the most recent
	if (good_state_a && good_state_b)
	{
		if (current_a > current_b)
			good_state_b = false;
		else
			good_state_a = false;
	}

        // Use data from the most recent state file
	if (good_state_a && !good_state_b)
	{
		current_p = current_a;
		primecount = primecount_a;
		checksum = checksum_a;
		state_a = false;
		if(thread_id == 0){
			last_trickle = trickle_a;
		}
		return 1;
	}
        if (good_state_b && !good_state_a)
        {
                current_p = current_b;
		primecount = primecount_b;
		checksum = checksum_b;
		state_a = true;
		if(thread_id == 0){
			last_trickle = trickle_b;
		}
		return 1;
        }

	// If we got here, neither state file was good
	return 0;
}


void checkpoint_thread( searchData sd, uint64_t my_P, int thread_id, bool & state_a, uint64_t primecount, uint64_t checksum, uint64_t & last_trickle )
{
	FILE *out;
	char buffer[20];

	if ( sprintf( buffer, "s%c%d.tmp", (state_a)?'A':'B', thread_id) < 0 ){
		fprintf(stderr,"error in sprintf()\n");
		exit(EXIT_FAILURE);
	}	

	if ((out = boinc_fopen(buffer,"w")) == NULL)
		fprintf(stderr,"Cannot open %s !!!\n",buffer);

	if (fprintf(out,"%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %u %" PRIu64 "\n",sd.workunit,my_P,primecount,checksum,sd.threads,last_trickle) < 0){
		fprintf(stderr,"Cannot write to %s !!! Continuing...\n",buffer);
		// Attempt to close, even though we failed to write
		fclose(out);
	}
	else{
		// If state file is closed OK, write to the other state file
		// next time around
		if (fclose(out) == 0) 
			state_a = !state_a; 
	}

	if(thread_id == 0){
		handle_trickle_up( last_trickle );
		boinc_checkpoint_completed();
	}

}


void clear_temp_factors_file( int thread_id ){

	char buffer[20];	

	if ( sprintf( buffer, "f%d.tmp",thread_id) < 0 ){
		fprintf(stderr,"error in sprintf()\n");
		exit(EXIT_FAILURE);
	}

	// clear result file
	FILE * temp_file = boinc_fopen(buffer,"w");
	if (temp_file == NULL){
		fprintf(stderr,"Cannot open %s !!!\n",buffer);
		exit(EXIT_FAILURE);
	}
	fclose(temp_file);

}


void report_solution_temp( char * results, int thread_id ){

	char buffer[20];	

	if ( sprintf( buffer, "f%d.tmp",thread_id) < 0 ){
		fprintf(stderr,"error in sprintf()\n");
		exit(EXIT_FAILURE);
	}	

	FILE * resfile = boinc_fopen(buffer,"a");

	if( resfile == NULL ){
		fprintf(stderr,"Cannot open %s !!!\n",buffer);
		exit(EXIT_FAILURE);
	}

	if( fprintf( resfile, "%s", results ) < 0 ){
		fprintf(stderr,"Cannot write to %s !!!\n",buffer);
		exit(EXIT_FAILURE);
	}

	fclose(resfile);

}


// find the log base 2 of a number.
int lg2(uint64_t v) {

#ifdef __GNUC__
	return 63-__builtin_clzll(v);
#else
	int r = 0; // r will be lg(v)
	while (v >>= 1)r++;
	return r;
#endif

}


void setupSearch(searchData & sd){

	if(sd.pmin == 0 || sd.pmax == 0){
		printf("-p and -P arguments are required\n");
		fprintf(stderr, "-p and -P arguments are required\n");
		exit(EXIT_FAILURE);
	}

	if(sd.nmin == 0 || sd.nmax == 0){
		printf("-n and -N arguments are required\n");
		fprintf(stderr, "-n and -N arguments are required\n");
		exit(EXIT_FAILURE);
	}

	if (sd.nmin > sd.nmax){
		printf("nmin <= nmax is required\n");
		fprintf(stderr, "nmin <= nmax is required\n");
		exit(EXIT_FAILURE);
	}

	if(sd.cw){

		if(sd.nmax >= sd.pmin){
			printf("nmax < pmin is required\n");
			fprintf(stderr, "nmax < pmin is required\n");
			exit(EXIT_FAILURE);
		}

		sd.kmax = sd.nmax;
		sd.kmin = sd.nmin;
	}
	else{

		if(sd.kmax == 0){
			printf("-K argument is required\n");
			fprintf(stderr, "-K argument is required\n");
			exit(EXIT_FAILURE);
		}

		if(sd.kmin > sd.kmax){
			printf("kmin <= kmax is required\n");
			fprintf(stderr, "kmin <= kmax is required\n");
			exit(EXIT_FAILURE);
		}

		if(sd.kmax >= sd.pmin){
			printf("kmax < pmin is required\n");
			fprintf(stderr, "kmax < pmin is required\n");
			exit(EXIT_FAILURE);
		}

		uint32_t b0 = 0, b1 = 0;
		b0 = sd.kmin/sd.kstep;
		b1 = sd.kmax/sd.kstep;
		sd.kmin = b0*sd.kstep+sd.koffset;
		sd.kmax = b1*sd.kstep+sd.koffset;
	}


	for (sd.nstep = 1; ( (uint64_t)(sd.kmax) << sd.nstep ) < sd.pmin; sd.nstep++);

	if((((uint64_t)1) << (64-sd.nstep)) > sd.pmin) {

		uint64_t pmin_1 = (((uint64_t)1) << (64-sd.nstep));

		printf("Error: pmin is not large enough (or nmax is close to nmin).\n");
		fprintf(stderr, "Error: pmin is not large enough (or nmax is close to nmin).\n");

		sd.pmin = sd.kmax + 1;
		for (sd.nstep = 1; ( (uint64_t)(sd.kmax) << sd.nstep ) < sd.pmin; sd.nstep++);

		while((((uint64_t)1) << (64-sd.nstep)) > sd.pmin) {
			sd.pmin *= 2;
			sd.nstep++;
		}
		if(pmin_1 < sd.pmin){
			sd.pmin = pmin_1;
		}

		printf("This program will work by the time pmin == %" PRIu64 ".\n", sd.pmin);
		fprintf(stderr, "This program will work by the time pmin == %" PRIu64 ".\n", sd.pmin);

		exit(EXIT_FAILURE);
	}

	if (sd.nstep > (sd.nmax-sd.nmin+1))
		sd.nstep = (sd.nmax-sd.nmin+1);

	// For TPS, decrease the ld_nstep by one to allow overlap, checking both + and -
	sd.nstep--;

	// Use the 32-step algorithm where useful.
	if(sd.nstep >= 32 && (((uint64_t)1) << 32) <= sd.pmin) {
		sd.nstep = 32;
	}

	// search twin, decrease by one to allow overlap, checking both + and -
	sd.nmin--;

	sd.bbits = lg2(sd.nmin);

	if(sd.bbits < 6) {
		printf("Error: nmin too small at %d (must be at least 65).\n", sd.nmin+1);
		fprintf(stderr, "Error: nmin too small at %d (must be at least 65).\n", sd.nmin+1);
		exit(EXIT_FAILURE);
	}

	// r = 2^-i * 2^64 (mod N), something that can be done in a uint64_t!
	// If i is large (and it should be at least >= 32), there's a very good chance no mod is needed!
	sd.r0 = ((uint64_t)1) << (64-(sd.nmin >> (sd.bbits-5)));

	sd.bbits = sd.bbits-6;

	sd.mont_nstep = 64-sd.nstep;

	// data for checksum
	uint32_t maxn;

	maxn = ( (sd.nmax-sd.nmin) / sd.nstep) * sd.nstep;
	maxn += sd.nmin;

	if( maxn < sd.nmax ){
		maxn += sd.nstep;
	}

	int bbits1 = lg2(maxn) - 5;
	sd.r1 = ((uint64_t)1) << (64-(maxn >> bbits1));
	--bbits1;
	sd.bbits1 = bbits1;
	sd.lastN = maxn;

	// for checkpoints
	sd.workunit = sd.pmin + sd.pmax + (uint64_t)sd.nmin + (uint64_t)sd.nmax + (uint64_t)sd.kmin + (uint64_t)sd.kmax;

	// for BOINC fraction done
	sd.wu_range = (double)(sd.pmax - sd.pmin);

	// result buffer size per thread
	sd.num_results = 1000000u / sd.threads;

}


void ckerr(int err){
	if(err){
		fprintf(stderr, "ERROR: pthreads, code: %d\n", err);
		exit(EXIT_FAILURE);
	}
}


bool goodfactor_cw(uint32_t uk, uint32_t n, int c){

	uint64_t k = uk;

	if(prime15[(uint32_t)(((k<<(n&3))+c)%15)] && (uint32_t)(((k<<(n%3))+c)%7) != 0)
		return true;

	return false;

}


bool goodfactor(uint32_t uk, uint32_t n, int c){

	uint64_t k = uk;
	uint64_t mod31;
	// Check that K*2^N+/-1 is not divisible by 3, 5, or 7, to minimize factors printed.
	// We do 3 and 5 at the same time (15 = 2^4-1), then 7 (=2^3-1).
	// Then 17, 11 (and 31), 13, and maybe 19, if there's space. 23 can also go in there, if it's worth it.
	// (k*(1<<(n%2))+c)%3 == 0
	if(	prime15[(uint32_t)(((k<<(n&3))+c)%15)] && 
		(uint32_t)(((k<<(n%3))+c)%7) != 0 &&
		(uint32_t)(((k<<(n&7))+c)%17) != 0 && 
		(uint32_t)((mod31=(k<<(n%10))+c)%11) != 0 &&
		(uint32_t)(((k<<(n%11))+c)%23) != 0 &&
		(uint32_t)(((k<<(n%12))+c)%13) != 0 &&
		(uint32_t)(((k<<(n%18))+c)%19) != 0 )
		if( (uint32_t)(mod31%31) != 0 )
			return true;

	return false;

}


void cpu_sieve( searchData & sd ){

	uint64_t range;

	sieve_small_primes(11);

	time_t totals, totalf;
	if(boinc_is_standalone()){
		time(&totals);
	}

	// setup parameters
	setupSearch(sd);

	fprintf(stderr, "Starting sieve at p: %" PRIu64 " n: %u k: %u\nStopping sieve at P: %" PRIu64 " N: %u K: %u\n", sd.pmin, sd.nmin+1, sd.kmin, sd.pmax, sd.nmax, sd.kmax);
	if(boinc_is_standalone()){
		printf("Starting sieve at p: %" PRIu64 " n: %u k: %u\nStopping sieve at P: %" PRIu64 " N: %u K: %u\n", sd.pmin, sd.nmin+1, sd.kmin, sd.pmax, sd.nmax, sd.kmax);
	}

	// clear final factor file
	FILE * temp_file = boinc_res_fopen("factors.txt","w");
	if (temp_file == NULL){
		fprintf(stderr,"Cannot open %s !!!\n","factors.txt");
		exit(EXIT_FAILURE);
	}
	fclose(temp_file);


	fprintf(stderr,"Starting search...\n");
	if(boinc_is_standalone()){
		printf("Starting search...\n");
		printf("nstep: %u\n",sd.nstep);
	}


	// initialize shared data
	g_complete = 0;
	g_primecount = 0;
	g_checksum = 0;

	// setup search range for each thread
	if( (sd.pmax - sd.pmin) % sd.threads == 0 ){
		range = (sd.pmax - sd.pmin) / sd.threads;
	}
	else{
		range = (sd.pmax - sd.pmin + sd.threads) / sd.threads;
	}

	// initialize pthread mutex
	ckerr(pthread_mutex_init(&lock1,NULL));
	ckerr(pthread_mutex_init(&lock2,NULL));

	pthread_t thr[sd.threads];
	thread_data_t thr_data[sd.threads];

	// create threads
	uint64_t currp = sd.pmin;

	for (uint32_t k = 0; k < sd.threads; ++k) {
		thr_data[k].id = k;
		thr_data[k].sd = sd;
		thr_data[k].pmin = currp;
		thr_data[k].pmax = currp + range;
		if(thr_data[k].pmax > sd.pmax) thr_data[k].pmax = sd.pmax;

		if(sd.avx512){
			if(sd.cw){
				ckerr(pthread_create(&thr[k], NULL, thr_func_cw_avx512, &thr_data[k]));
			}
			else{
				ckerr(pthread_create(&thr[k], NULL, thr_func_avx512, &thr_data[k]));
			}
		}
		else{
			if(sd.cw){
				ckerr(pthread_create(&thr[k], NULL, thr_func_cw, &thr_data[k]));
			}
			else{
				ckerr(pthread_create(&thr[k], NULL, thr_func, &thr_data[k]));
			}
		}

		currp += range;
	}


	// block until all threads complete
	for (uint32_t k = 0; k < sd.threads; ++k) {
		ckerr(pthread_join(thr[k], NULL));
	}

	boinc_fraction_done(1.0);

	if(boinc_is_standalone()) printf("Tests done: 100.0%%\n");

	boinc_begin_critical_section();

	gatherfactors(sd);
	sd.primecount = g_primecount;
	sd.checksum = g_checksum;

	fprintf(stderr,"Search complete.\nfactors %u, prime count %" PRIu64 "\n", sd.factorcount, sd.primecount);

	boinc_end_critical_section();

	if(boinc_is_standalone()){
		time(&totalf);
		printf("Search finished in %d sec.\n", (int)totalf - (int)totals);
		printf("factors %u, prime count %" PRIu64 ", checksum %016" PRIX64 "\n", sd.factorcount, sd.primecount, sd.checksum);
	}

	ckerr(pthread_mutex_destroy(&lock1));
	ckerr(pthread_mutex_destroy(&lock2));

	small_primes_free();
}


void run_test( searchData & sd ){

	int goodtest = 0;

	printf("Beginning self test of 4 ranges.\n");

//	-p 25636026e6 -P 25636030e6 -n 10000000 -N 25000000 -c		nstep 19
	sd.pmin = 25636026000000;
	sd.pmax = 25636030000000;
	sd.nmin = 10000000;
	sd.nmax = 25000000;
	sd.kmin = 0;
	sd.kmax = 0;
	sd.cw = true;
	cpu_sieve( sd );
	if( sd.factorcount == 2 && sd.primecount == 129869 && sd.checksum == 0x4544591DC69ACD83 ){
		printf("CW test case 1 passed.\n\n");
		fprintf(stderr,"CW test case 1 passed.\n");
		++goodtest;
	}
	else{
		printf("CW test case 1 failed.\n\n");
		fprintf(stderr,"CW test case 1 failed.\n");
	}
	sd.checksum = 0;
	sd.primecount = 0;
	sd.factorcount = 0;

//	-p 556439300e6 -P 556439440e6 -n 100 -N 100000 -c		nstep 32
	sd.pmin = 556439300000000;
	sd.pmax = 556439440000000;
	sd.nmin = 100;
	sd.nmax = 100000;
	sd.kmin = 0;
	sd.kmax = 0;
	sd.cw = true;
	cpu_sieve( sd );
	if( sd.factorcount == 1 && sd.primecount == 4123452 && sd.checksum == 0x8FEC30979896A3C0 ){
		printf("CW test case 2 passed.\n\n");
		fprintf(stderr,"CW test case 2 passed.\n");
		++goodtest;
	}
	else{
		printf("CW test case 2 failed.\n\n");
		fprintf(stderr,"CW test case 2 failed.\n");
	}
	sd.checksum = 0;
	sd.primecount = 0;
	sd.factorcount = 0;


//	-p838338347800e6 -P838338347820e6 -k5 -K9999 -n6000000 -N9000000	nstep 32
	sd.pmin = 838338347800000000;
	sd.pmax = 838338347820000000;
	sd.nmin = 6000000;
	sd.nmax = 9000000;
	sd.kmin = 5;
	sd.kmax = 9999;
	sd.cw = false;
	cpu_sieve( sd );
	if( sd.factorcount == 1 && sd.primecount == 484024 && sd.checksum == 0xA7DC855BCB311759 ){
		printf("test case 3 passed.\n\n");
		fprintf(stderr,"test case 3 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 3 failed.\n\n");
		fprintf(stderr,"test case 3 failed.\n");
	}
	sd.checksum = 0;
	sd.primecount = 0;
	sd.factorcount = 0;

//	-p42070000e6 -P42070050e6 -k 1201 -K 9999 -n 100 -N 2000000		nstep 31
	sd.pmin = 42070000000000;
	sd.pmax = 42070050000000;
	sd.nmin = 100;
	sd.nmax = 2000000;
	sd.kmin = 1201;
	sd.kmax = 9999;
	sd.cw = false;
	cpu_sieve( sd );
	if( sd.factorcount == 70 && sd.primecount == 1592285 && sd.checksum == 0x727796B2D3677937 ){
		printf("test case 4 passed.\n\n");
		fprintf(stderr,"test case 4 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 4 failed.\n\n");
		fprintf(stderr,"test case 4 failed.\n");
	}



	if(goodtest == 4){
		printf("All test cases completed successfully!\n");
		fprintf(stderr, "All test cases completed successfully!\n");
	}
	else{
		printf("Self test FAILED!\n");
		fprintf(stderr, "Self test FAILED!\n");
	}

}

