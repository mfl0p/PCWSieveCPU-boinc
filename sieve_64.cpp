/*
	PCWSieve - 64 bit
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

/*  BEGIN REDC CODE  */

// Compute T=a<<s; m = (T*Ns)%2^64; T += m*N; if (T>N) T-= N;
// rax is passed in as a * Ns.
static uint64_t shiftmod_REDC (const uint64_t a, const uint64_t N, uint64_t rax, const uint32_t mont_nstep, const uint32_t nstep){

	uint64_t rcx;
	unsigned __int128 res;

	rax <<= mont_nstep; // So this is a*Ns*(1<<s) == (a<<s)*Ns.
	rcx = a >> nstep;
	rcx += (rax!=0)?1:0;
	res = (unsigned __int128)rax * N;
	rax = res >> 64;
	rax += rcx;
	rcx = rax - N;
	rax = (rax>N)?rcx:rax;

	return rax;
}


// mulmod_REDC(1, 1, N, Ns)
// But note that mulmod_REDC(a, 1, N, Ns) == mulmod_REDC(1, 1, N, Ns*a).
static uint64_t onemod_REDC(const uint64_t N, uint64_t rax) {

	uint64_t rcx;
	unsigned __int128 res;

	// Akruppa's way, Compute T=a*b; m = (T*Ns)%2^64; T += m*N; if (T>N) T-= N;
	rcx = (rax!=0)?1:0;

	res = (unsigned __int128)rax * N;
	rax = res >> 64;
	rax += rcx;

	rcx = rax - N;
	rax = (rax>N)?rcx:rax;

	return rax;
}

// Like mulmod_REDC(a, 1, N, Ns) == mulmod_REDC(1, 1, N, Ns*a).
static uint64_t mod_REDC(const uint64_t a, const uint64_t N, const uint64_t Ns) {
	return onemod_REDC(N, Ns*a);
}


static uint64_t mulmod_REDC (const uint64_t a, const uint64_t b, const uint64_t N, const uint64_t Ns){

	uint64_t rax, rcx;
	unsigned __int128 res;

	// Akruppa's way, Compute T=a*b; m = (T*Ns)%2^64; T += m*N; if (T>N) T-= N;
	res = (unsigned __int128)a * b;
	rax = (uint64_t)res;
	rcx = res >> 64;
  
        rax *= Ns;
        rcx += ( (rax != 0)?1:0 );

	res = (unsigned __int128)rax * N;
	rax = res >> 64;
	rax += rcx;

        rcx = rax - N;
        rax = (rax>N)?rcx:rax;

        return rax;
}


// Hybrid powmod, sidestepping several loops and possible mispredicts, and with no longmod!
/* Compute (2^-1)^b (mod m), using Montgomery arithmetic. */
static uint64_t invpowmod_REDClr (const uint64_t N, const uint64_t Ns, const uint32_t l_nmin, uint64_t r, int bbits) {

	// Now work through the other bits of nmin.
	for(; bbits >= 0; --bbits) {
		// Just keep squaring r.
		r = mulmod_REDC(r, r, N, Ns);
		// If there's a one bit here, multiply r by 2^-1 (aka divide it by 2 mod N).
		if(l_nmin & (1u << bbits)) {
			r += (r&1)?N:0;
			r >>= 1;
		}
	}

	// Convert back to standard.
	r = mod_REDC (r, N, Ns);

	return r;
}


static uint64_t invmod2pow_ul (const uint64_t n){

	uint64_t r;
	const uint32_t in = (uint32_t)n;

	// Suggestion from PLM: initing the inverse to (3*n) XOR 2 gives the
	// correct inverse modulo 32, then 3 (for 32 bit) or 4 (for 64 bit) 
	// Newton iterations are enough.
	r = (n+n+n) ^ ((uint64_t)2);
	// Newton iteration
	r += r - (uint32_t) r * (uint32_t) r * in;
	r += r - (uint32_t) r * (uint32_t) r * in;
	r += r - (uint32_t) r * (uint32_t) r * in;
	r += r - r * r * n;

	return r;
}


void get_results(int64_t * fP, uint32_t * fK, uint32_t * fN, uint32_t count, int thread, bool cw){

	char buffer[256];

	// print factors to temporary file and checkpoint
	char * resbuff = (char *)malloc( count * sizeof(char) * 256 );
	if( resbuff == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}
	resbuff[0] = '\0';

	for(uint32_t m=0; m < count; ++m){

		uint64_t p = (fP[m] < 0)?-fP[m]:fP[m];
		int s = (fP[m] < 0)?-1:1;

		if(cw){
			if( sprintf( buffer, "%" PRIu64 " | %u*2^%u%+d\n",p,fN[m],fN[m],s) < 0 ){
				fprintf(stderr,"error in sprintf()\n");
				exit(EXIT_FAILURE);
			}
		}
		else{
			if( sprintf( buffer, "%" PRIu64 " | %u*2^%u%+d\n",p,fK[m],fN[m],s) < 0 ){
				fprintf(stderr,"error in sprintf()\n");
				exit(EXIT_FAILURE);
			}
		}

		strcat( resbuff, buffer );
	}

	report_solution_temp( resbuff, thread );
	free(resbuff);

}


void *thr_func_cw(void *arg) {

	thread_data_t *data = (thread_data_t *)arg;
	bool state_a = true;
	uint64_t checksum = 0;
	uint64_t primecount = 0;
	uint32_t factorcount = 0;
	time_t ckpt_curr, ckpt_last, boinc_last, boinc_curr;
	primesieve_iterator it;
	uint64_t my_P;
	uint64_t current_p;
	uint64_t last_P;
	uint64_t complete;
	uint64_t last_trickle = 0;

	// for fraction done
	last_P = data->pmin;

	if(data->sd.test){
		current_p = data->pmin;
		clear_temp_factors_file(data->id);
	}
	else{
		// Resume from checkpoint if there is one
		if( read_state_thread( data->sd, current_p, data->id, state_a, primecount, checksum, last_trickle ) ){
			if(data->id == 0){
				if(boinc_is_standalone()){
					printf("Resuming search from checkpoint.\n");
				}
				fprintf(stderr,"Resuming search from checkpoint.\n");
			}
		}
		// starting from beginning
		else{
			current_p = data->pmin;
			clear_temp_factors_file(data->id);

			// setup boinc trickle up
			if(data->id == 0){
				last_trickle = (uint64_t)time(NULL);
			}
		}
	}

	// initialize prime generator.  add prime gap to max so iterator won't reset at end of range.
	primesieve_init(&it);
	primesieve_jump_to(&it, current_p, data->pmax+1509);

	// buffer factors between checkpoints
	int64_t * factorP = (int64_t *)malloc( data->sd.num_results * sizeof(int64_t));
	if( factorP == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}
	uint32_t * factorN = (uint32_t *)malloc( data->sd.num_results * sizeof(uint32_t));
	if( factorN == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}

	time(&boinc_last);
	time(&ckpt_last);

	while(true){

		my_P = primesieve_next_prime(&it);
		if(my_P >= data->pmax){
			break;
		}

		// update BOINC fraction done every 5 sec
		time(&boinc_curr);
		if( ((int)boinc_curr - (int)boinc_last) > 4 ){
			boinc_last = boinc_curr;
			complete = my_P - last_P;

			ckerr(pthread_mutex_lock(&lock2));
			g_complete += complete;
			if(data->id == 0){
				// update BOINC fraction done if thread 0
				double fd = (double)(g_complete)/data->sd.wu_range;
				if(boinc_is_standalone()) printf("Tests done: %.1f%%\n",fd*100.0);
				boinc_fraction_done(fd);
			}
			ckerr(pthread_mutex_unlock(&lock2));
	
			last_P = my_P;
		}

		// 1 minute checkpoint
		time(&ckpt_curr);
		if( ((int)ckpt_curr - (int)ckpt_last) > 60 ){
			ckpt_last = ckpt_curr;

			boinc_begin_critical_section();

			if(factorcount > 0){
				get_results(factorP, factorN, factorN, factorcount, data->id, data->sd.cw);
				factorcount = 0;
			}

			checkpoint_thread( data->sd, my_P, data->id, state_a, primecount, checksum, last_trickle );

			boinc_end_critical_section();
		}

		// setup
		uint64_t Ps = -invmod2pow_ul(my_P); /* Ns = -N^{-1} % 2^64 */

		// Calculate k0, in Montgomery form.
		uint64_t k0 = invpowmod_REDClr(my_P, Ps, data->sd.nmin, data->sd.r0, data->sd.bbits);

		uint32_t n = data->sd.nmin;

		do { 	// Remaining steps are all of equal size nstep

			// Select the even one.
			uint64_t kpos = (k0 & 1)?(my_P - k0):k0;

			uint32_t i = __builtin_ctzll(kpos);

			if(i <= data->sd.nstep) {
				uint64_t the_k = kpos >> i;
				uint64_t the_n = n + i;
				if(the_k <= the_n){
					while(the_k < the_n){
						the_k <<= 1;
						the_n--;
					}
					if(the_k == the_n && the_n <= data->sd.nmax) {
						int s = (kpos==k0)?-1:1;
						if( goodfactor_cw(the_k, the_n, s)){
							if(try_all_factors(the_k, the_n, s) == 0){	// check for a small prime factor of the number
								// check the factor actually divides the number
								if(verify_factor(my_P, the_k, the_n, s)){
//									printf("cw factor found, p %" PRIu64 " k %u n %u s %d\n",my_P,the_k,the_n,s);
									factorP[factorcount] = (s==1) ? (int64_t)my_P : -((int64_t)my_P);
									factorN[factorcount] = the_n;
									++factorcount;
									if(factorcount > data->sd.num_results){
										printf("ERROR: result array overflow!\n");
										fprintf(stderr,"ERROR: result array overflow!\n");
										exit(EXIT_FAILURE);
									}
									// add factor to checksum
									checksum += the_n*2;
									(s == 1)?(++checksum):(--checksum);					
								}
								else{
									printf("ERROR: found invalid factor!\n");
									fprintf(stderr,"ERROR: found invalid factor!\n");
									exit(EXIT_FAILURE);
								}
							}
						}
					}
				}
			}

			// Proceed to the K for the next N.
			n += data->sd.nstep;
			k0 = shiftmod_REDC(k0, my_P, k0*Ps, data->sd.mont_nstep, data->sd.nstep);

		} while (n < data->sd.nmax);

		// calculate k for last value of N, for checksum.
		uint64_t k1 = invpowmod_REDClr(my_P, Ps, data->sd.lastN, data->sd.r1, data->sd.bbits1);

		if(k0 != k1){
			printf("ERROR: checksum calculation in thread %d!\n",data->id);
			fprintf(stderr,"ERROR: checksum calculation in thread %d!\n",data->id);
			exit(EXIT_FAILURE);
		}

		++primecount;
		checksum += k0 + my_P;
	}

	// final checkpoint
	boinc_begin_critical_section();

	if(factorcount > 0){
		get_results(factorP, factorN, factorN, factorcount, data->id, data->sd.cw);
		factorcount = 0;
	}

	checkpoint_thread( data->sd, my_P, data->id, state_a, primecount, checksum, last_trickle );

	boinc_end_critical_section();

	// update global primecount and checksum
	ckerr(pthread_mutex_lock(&lock1));
	g_primecount += primecount;
	g_checksum += checksum;
	ckerr(pthread_mutex_unlock(&lock1));

	// update fraction done
	complete = my_P - last_P;
	ckerr(pthread_mutex_lock(&lock2));
	g_complete += complete;
	ckerr(pthread_mutex_unlock(&lock2));

	primesieve_free_iterator(&it);

	free(factorP);
	free(factorN);

	pthread_exit(NULL);

	return NULL;
}


void *thr_func(void *arg) {

	thread_data_t *data = (thread_data_t *)arg;
	bool state_a = true;
	uint64_t checksum = 0;
	uint64_t primecount = 0;
	uint32_t factorcount = 0;
	time_t ckpt_curr, ckpt_last, boinc_last, boinc_curr;
	primesieve_iterator it;
	uint64_t my_P;
	uint64_t current_p;
	uint64_t last_P;
	uint64_t complete;
	uint64_t last_trickle = 0;

	// for fraction done
	last_P = data->pmin;

	if(data->sd.test){
		current_p = data->pmin;
		clear_temp_factors_file(data->id);
	}
	else{
		// Resume from checkpoint if there is one
		if( read_state_thread( data->sd, current_p, data->id, state_a, primecount, checksum, last_trickle ) ){
			if(data->id == 0){
				if(boinc_is_standalone()){
					printf("Resuming search from checkpoint.\n");
				}
				fprintf(stderr,"Resuming search from checkpoint.\n");
			}
		}
		// starting from beginning
		else{
			current_p = data->pmin;
			clear_temp_factors_file(data->id);

			// setup boinc trickle up
			if(data->id == 0){
				last_trickle = (uint64_t)time(NULL);
			}
		}
	}

	// initialize prime generator.  add prime gap to max so iterator won't reset at end of range.
	primesieve_init(&it);
	primesieve_jump_to(&it, current_p, data->pmax+1509);

	// buffer factors between checkpoints
	int64_t * factorP = (int64_t *)malloc( data->sd.num_results * sizeof(int64_t));
	if( factorP == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}
	uint32_t * factorN = (uint32_t *)malloc( data->sd.num_results * sizeof(uint32_t));
	if( factorN == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}
	uint32_t * factorK = (uint32_t *)malloc( data->sd.num_results * sizeof(uint32_t));
	if( factorK == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}

	time(&boinc_last);
	time(&ckpt_last);

	while(true){

		my_P = primesieve_next_prime(&it);
		if(my_P >= data->pmax){
			break;
		}

		// update BOINC fraction done every 5 sec
		time(&boinc_curr);
		if( ((int)boinc_curr - (int)boinc_last) > 4 ){
			boinc_last = boinc_curr;
			complete = my_P - last_P;

			ckerr(pthread_mutex_lock(&lock2));
			g_complete += complete;
			if(data->id == 0){
				// update BOINC fraction done if thread 0
				double fd = (double)(g_complete)/data->sd.wu_range;
				if(boinc_is_standalone()) printf("Tests done: %.1f%%\n",fd*100.0);
				boinc_fraction_done(fd);
			}
			ckerr(pthread_mutex_unlock(&lock2));
	
			last_P = my_P;
		}

		// 1 minute checkpoint
		time(&ckpt_curr);
		if( ((int)ckpt_curr - (int)ckpt_last) > 60 ){
			ckpt_last = ckpt_curr;

			boinc_begin_critical_section();

			if(factorcount > 0){
				get_results(factorP, factorK, factorN, factorcount, data->id, data->sd.cw);
				factorcount = 0;
			}

			checkpoint_thread( data->sd, my_P, data->id, state_a, primecount, checksum, last_trickle );

			boinc_end_critical_section();
		}

		// setup
		uint64_t Ps = -invmod2pow_ul(my_P); /* Ns = -N^{-1} % 2^64 */

		// Calculate k0, in Montgomery form.
		uint64_t k0 = invpowmod_REDClr(my_P, Ps, data->sd.nmin, data->sd.r0, data->sd.bbits);

		uint32_t n = data->sd.nmin;

		do { 	// Remaining steps are all of equal size nstep

			// Select the even one.
			uint64_t kpos = (k0 & 1)?(my_P - k0):k0;

			uint32_t i = __builtin_ctzll(kpos);

			if(i <= data->sd.nstep) {
				uint64_t the_k = kpos >> i;
				if(the_k <= 0xffffffff){
					uint32_t the_n = n + i;
					if(the_k <= data->sd.kmax && the_k >= data->sd.kmin && the_n <= data->sd.nmax) {
						int s = (kpos==k0)?-1:1;
						if( goodfactor(the_k, the_n, s)){
							if(try_all_factors(the_k, the_n, s) == 0){	// check for a small prime factor of the number
								// check the factor actually divides the number
								if(verify_factor(my_P, the_k, the_n, s)){
//									printf("factor found, p %" PRIu64 " k %u n %u s %d\n",my_P,the_k,the_n,s);
									factorP[factorcount] = (s==1) ? (int64_t)my_P : -((int64_t)my_P);
									factorN[factorcount] = the_n;
									factorK[factorcount] = the_k;
									++factorcount;
									if(factorcount > data->sd.num_results){
										printf("ERROR: result array overflow!\n");
										fprintf(stderr,"ERROR: result array overflow!\n");
										exit(EXIT_FAILURE);
									}
									// add factor to checksum
									checksum += the_n;
									checksum += the_k;
									(s == 1)?(++checksum):(--checksum);					
								}
								else{
									printf("ERROR: found invalid factor!\n");
									fprintf(stderr,"ERROR: found invalid factor!\n");
									exit(EXIT_FAILURE);
								}
							}
						}
					}
				}
			}

			// Proceed to the K for the next N.
			n += data->sd.nstep;
			k0 = shiftmod_REDC(k0, my_P, k0*Ps, data->sd.mont_nstep, data->sd.nstep);

		} while (n < data->sd.nmax);

		// calculate k for last value of N, for checksum.
		uint64_t k1 = invpowmod_REDClr(my_P, Ps, data->sd.lastN, data->sd.r1, data->sd.bbits1);

		if(k0 != k1){
			printf("ERROR: checksum calculation in thread %d!\n",data->id);
			fprintf(stderr,"ERROR: checksum calculation in thread %d!\n",data->id);
			exit(EXIT_FAILURE);
		}

		++primecount;
		checksum += k0 + my_P;
	}

	// final checkpoint
	boinc_begin_critical_section();

	if(factorcount > 0){
		get_results(factorP, factorK, factorN, factorcount, data->id, data->sd.cw);
		factorcount = 0;
	}

	checkpoint_thread( data->sd, my_P, data->id, state_a, primecount, checksum, last_trickle );

	boinc_end_critical_section();

	// update global primecount and checksum
	ckerr(pthread_mutex_lock(&lock1));
	g_primecount += primecount;
	g_checksum += checksum;
	ckerr(pthread_mutex_unlock(&lock1));

	// update fraction done
	complete = my_P - last_P;
	ckerr(pthread_mutex_lock(&lock2));
	g_complete += complete;
	ckerr(pthread_mutex_unlock(&lock2));

	primesieve_free_iterator(&it);

	free(factorP);
	free(factorN);
	free(factorK);

	pthread_exit(NULL);

	return NULL;
}


