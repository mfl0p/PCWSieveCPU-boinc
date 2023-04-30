/*
	PCWSieve - AVX512
	Bryan Little, March 6, 2023
	
	Search algorithm by
	Geoffrey Reynolds, 2009
	Ken Brazier, 2009
	https://github.com/Ken-g6/PSieve-CUDA/tree/redcl
	https://github.com/Ken-g6/PSieve-CUDA/tree/cw

	With contributions by
	Yves Gallot

*/
#include <immintrin.h>

#include <unistd.h>
#include <pthread.h>

#include "boinc_api.h"

#include "primesieve.h"
#include "factor_proth.h"
#include "verify_factor.h"
#include "putil.h"
#include "cpu_sieve.h"

using namespace std; 


inline void __umul64wide(__m512i x, __m512i y, __m512i & lo, __m512i & hi) {   
 
	__m512i lomask = _mm512_set1_epi64(0xffffffff);

	__m512i xh     = _mm512_shuffle_epi32(x, _MM_PERM_CDAB 	);    // x0l, x0h, x1l, x1h
	__m512i yh     = _mm512_shuffle_epi32(y, _MM_PERM_CDAB 	);    // y0l, y0h, y1l, y1h

	__m512i w0     = _mm512_mul_epu32(x,  y);          // x0l*y0l, x1l*y1l
	__m512i w1     = _mm512_mul_epu32(x,  yh);         // x0l*y0h, x1l*y1h
	__m512i w2     = _mm512_mul_epu32(xh, y);          // x0h*y0l, x1h*y0l
	__m512i w3     = _mm512_mul_epu32(xh, yh);         // x0h*y0h, x1h*y1h

	__m512i w0l    = _mm512_and_epi64(w0, lomask);
	__m512i w0h    = _mm512_srli_epi64(w0, 32);

	__m512i s1     = _mm512_add_epi64(w1, w0h);
	__m512i s1l    = _mm512_and_epi64(s1, lomask);
	__m512i s1h    = _mm512_srli_epi64(s1, 32);

	__m512i s2     = _mm512_add_epi64(w2, s1l);
	__m512i s2l    = _mm512_slli_epi64(s2, 32);
	__m512i s2h    = _mm512_srli_epi64(s2, 32);

	__m512i hi1    = _mm512_add_epi64(w3, s1h);
		hi1    = _mm512_add_epi64(hi1, s2h);

	__m512i lo1    = _mm512_add_epi64(w0l, s2l);

	hi = hi1;
	lo = lo1;
}


// Same function for nstep < 32. (SMall.)
// Third argument must be passed in as only the low register, as we're effectively left-shifting 32 plus a small number.
inline __m512i shiftmod_REDCsm_avx512 (__m512i rcx, const __m512i N, __m512i rax, const uint32_t sm_mont_nstep, const uint32_t nstep)
{
	__m512i x;
	__mmask8 m;

	//	rax <<= (mont_nstep-32);
	rax	= _mm512_slli_epi64(rax, sm_mont_nstep);
	//	chop hi bits
	rax	= _mm512_and_epi64(rax, _mm512_set1_epi64(0xffffffff));

	//	rcx >>= nstep;
	rcx	= _mm512_srli_epi64(rcx, nstep);

	//	rcx += (ulong)(mad_hi(rax, (uint)N, (uint)((rax!=0)?1:0) ) );
	x	= _mm512_mul_epu32(rax, N);
	x	= _mm512_srli_epi64(x, 32);
	rcx	= _mm512_add_epi64(rcx, x);
	m	= _mm512_cmpneq_epi64_mask(rax, _mm512_setzero_si512());
	rcx	= _mm512_mask_add_epi64(rcx, m, rcx, _mm512_set1_epi64(1));

	//	rcx += mul_wide_u32(rax, (uint)(N>>32));
	x	= _mm512_srli_epi64(N, 32);
	x	= _mm512_mul_epu32(rax, x);
	rcx	= _mm512_add_epi64(rcx, x);

	//	rcx = (rcx>N)?(rcx-N):rcx;
	m	= _mm512_cmpgt_epi64_mask(rcx, N);
	rcx	= _mm512_mask_sub_epi64(rcx, m, rcx, N);

	return rcx;
}


// Same function, for a constant NSTEP of 32.
// Third argument must be passed in as only the low register
inline __m512i shiftmod_REDC32_avx512 (__m512i rcx, const __m512i N, __m512i rax)
{
	__m512i x;
	__mmask8 m;

	//	chop hi bits
	rax	= _mm512_and_epi64(rax, _mm512_set1_epi64(0xffffffff));

	//	rcx >>= 32;
	rcx	= _mm512_srli_epi64(rcx, 32);

	//	rcx += mad_hi( rax, (uint)N, (uint)((rax!=0)?1:0) );
	x	= _mm512_mul_epu32(rax, N);
	x	= _mm512_srli_epi64(x, 32);
	rcx	= _mm512_add_epi64(rcx, x);
	m	= _mm512_cmpneq_epi64_mask(rax, _mm512_setzero_si512());
	rcx	= _mm512_mask_add_epi64(rcx, m, rcx, _mm512_set1_epi64(1));

	//	rcx = mad_wide_u32((rax),((uint)(N>>32)), rcx);
	x	= _mm512_srli_epi64(N, 32);
	x	= _mm512_mul_epu32(rax, x);
	rcx	= _mm512_add_epi64(rcx, x);

	//	rcx = (rcx>N)?(rcx-N):rcx;
	m	= _mm512_cmpgt_epi64_mask(rcx, N);
	rcx	= _mm512_mask_sub_epi64(rcx, m, rcx, N);

	return rcx;
}


// Compute T=a<<s; m = (T*Ns)%2^64; T += m*N; if (T>N) T-= N;
// rax is passed in as a * Ns.
inline __m512i shiftmod_REDC_avx512 (const __m512i a, const __m512i N, __m512i rax, const uint32_t mont_nstep, const uint32_t nstep){

	__m512i rcx, lo;
	__mmask8 m;

	//	rax <<= mont_nstep; // So this is a*Ns*(1<<s) == (a<<s)*Ns.
	rax	= _mm512_slli_epi64(rax, mont_nstep);

	//	rcx = a >> nstep;
	rcx	= _mm512_srli_epi64(a, nstep);

	//	rcx += (rax!=0)?1:0;
	m	= _mm512_cmpneq_epi64_mask(rax, _mm512_setzero_si512());
	rcx	= _mm512_mask_add_epi64(rcx, m, rcx, _mm512_set1_epi64(1));

	//	res = (unsigned __int128)rax * N;
	//	rax = res >> 64;
	__umul64wide(rax, N, lo, rax);

	//	rax += rcx;
	rax = _mm512_add_epi64(rax, rcx);

	//	rcx = rax - N;
	rcx = _mm512_sub_epi64(rax, N);

	//	rax = (rax>N)?rcx:rax;
	m	= _mm512_cmpgt_epi64_mask(rax, N);
	rax	= _mm512_mask_blend_epi64(m, rax, rcx);

	return rax;
}


inline __m512i mulmod_REDC_avx512 (const __m512i a, const __m512i b, const __m512i N, const __m512i Ns){

	__m512i rax, rcx, x;
	__mmask8 m;

	//	res = (unsigned __int128)a * b;
	//	rax = (uint64_t)res;
	//	rcx = res >> 64;
	__umul64wide(a,b,rax,rcx);

	//      rax *= Ns;
	rax	= _mm512_mullo_epi64(rax, Ns);

	//      rcx += ( (rax != 0)?1:0 );
	m	= _mm512_cmpneq_epi64_mask(rax, _mm512_setzero_si512());
	rcx	= _mm512_mask_add_epi64(rcx, m, rcx, _mm512_set1_epi64(1));

	//	res = (unsigned __int128)rax * N;
	//	rax = res >> 64;
	//	rax += rcx;
	__umul64wide(rax, N, x, rax);
	rax	= _mm512_add_epi64(rax, rcx);

	//      rcx = rax - N;
	rcx	= _mm512_sub_epi64(rax, N);

	//      rax = (rax>N)?rcx:rax;
	m	= _mm512_cmpgt_epi64_mask(rax,N);
	rax	= _mm512_mask_blend_epi64(m, rax, rcx);

        return rax;
}


// mulmod_REDC(1, 1, N, Ns)
// But note that mulmod_REDC(a, 1, N, Ns) == mulmod_REDC(1, 1, N, Ns*a).
inline __m512i onemod_REDC_avx512(const __m512i N, __m512i rax) {

	__m512i rcx, lo;
	__mmask8 m;

	//	rcx = (rax!=0)?1:0;
	m	= _mm512_cmpneq_epi64_mask(rax, _mm512_setzero_si512());
	rcx	= _mm512_mask_blend_epi64(m, _mm512_setzero_si512(), _mm512_set1_epi64(1));

	//	res = (unsigned __int128)rax * N;
	//	rax = res >> 64;
	//	rax += rcx;
	__umul64wide(rax, N, lo, rax);
	rax	= _mm512_add_epi64(rax, rcx);

	//      rcx = rax - N;
	rcx	= _mm512_sub_epi64(rax, N);

	//      rax = (rax>N)?rcx:rax;
	m	= _mm512_cmpgt_epi64_mask(rax,N);
	rax	= _mm512_mask_blend_epi64(m, rax, rcx);


	return rax;
}

// Like mulmod_REDC(a, 1, N, Ns) == mulmod_REDC(1, 1, N, Ns*a).
inline __m512i mod_REDC_avx512(const __m512i a, const __m512i N, const __m512i Ns) {
//	return onemod_REDC_avx512(N, Ns*a);
	return onemod_REDC_avx512(N, _mm512_mullo_epi64(Ns, a));
}


inline __m512i invpowmod_REDClr_avx512 (const __m512i N, const __m512i Ns, const uint32_t l_nmin, const __m512i r0, int bbits) {

	__m512i r = r0;
	__m512i x;
	__mmask8 m;
	const __m512i ONE = _mm512_set1_epi64(1);

	// Now work through the other bits of nmin.
	for(; bbits >= 0; --bbits) {
		// Just keep squaring r.
		r = mulmod_REDC_avx512(r, r, N, Ns);
		// If there's a one bit here, multiply r by 2^-1 (aka divide it by 2 mod N).
		if(l_nmin & (1u << bbits)) {
//			r += (r&1)?N:0;
			x = _mm512_and_epi64(r, ONE);
			m = _mm512_cmpeq_epi64_mask(x, ONE);
			x = _mm512_mask_blend_epi64(m, _mm512_setzero_si512(), N);
			r = _mm512_add_epi64(r, x);

//			r >>= 1;
			r = _mm512_srli_epi64(r, 1);
		}
	}

	// Convert back to standard.
	r = mod_REDC_avx512 (r, N, Ns);

	return r;
}


inline __m512i neg_invmod2pow_ul_avx512 (const __m512i n){

	__m512i r, rlo, in, lomask, x;

//	const uint32_t in = (uint32_t)n;
	lomask = _mm512_set1_epi64(0xffffffff);
	in     = _mm512_and_epi64(n, lomask); 

	// Suggestion from PLM: initing the inverse to (3*n) XOR 2 gives the
	// correct inverse modulo 32, then 3 (for 32 bit) or 4 (for 64 bit) 
	// Newton iterations are enough.
//	r = (n+n+n) ^ ((uint64_t)2);
	r = _mm512_add_epi64(n, n);
	r = _mm512_add_epi64(r, n);
	r = _mm512_xor_epi64(r, _mm512_set1_epi64(2));

	// Newton iteration
//	r += r - (uint32_t) r * (uint32_t) r * in;
	rlo = _mm512_and_epi64(r, lomask);
	x = _mm512_mul_epu32(rlo, rlo);
	x = _mm512_mul_epu32(x, in);
	x = _mm512_sub_epi64(r, x);
	r = _mm512_add_epi64(r, x);

//	r += r - (uint32_t) r * (uint32_t) r * in;
	rlo = _mm512_and_epi64(r, lomask);
	x = _mm512_mul_epu32(rlo, rlo);
	x = _mm512_mul_epu32(x, in);
	x = _mm512_sub_epi64(r, x);
	r = _mm512_add_epi64(r, x);

//	r += r - (uint32_t) r * (uint32_t) r * in;
	rlo = _mm512_and_epi64(r, lomask);
	x = _mm512_mul_epu32(rlo, rlo);
	x = _mm512_mul_epu32(x, in);
	x = _mm512_sub_epi64(r, x);
	r = _mm512_add_epi64(r, x);

//	r += r - r * r * n;
	x = _mm512_mullo_epi64(r, r);
	x = _mm512_mullo_epi64(x, n);
	x = _mm512_sub_epi64(r, x);
	r = _mm512_add_epi64(r, x);

	// negate
	r = _mm512_sub_epi64(_mm512_setzero_si512(), r);

	return r;
}


void get_results_avx512(int64_t * fP, uint32_t * fK, uint32_t * fN, uint32_t count, int thread, bool cw){

	char buffer[256];

	// sort results by prime size if needed
	// I found the compiler unrolls a loop and primes can be out of order
	if(count > 1){
		for (uint32_t i = 0; i < count-1; i++){    
			for (uint32_t j = 0; j < count-i-1; j++){
				uint64_t a = (fP[j]<0)?-fP[j]:fP[j];
				uint64_t b = (fP[j+1]<0)?-fP[j+1]:fP[j+1];
				if (a > b){
					swap(fP[j], fP[j+1]);
					swap(fN[j], fN[j+1]);
					if(!cw)	swap(fK[j], fK[j+1]);
				}
			}
		}
	}

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


void *thr_func_cw_avx512(void *arg) {

	thread_data_t *data = (thread_data_t *)arg;
	bool state_a = true;
	bool done = false;
	uint64_t checksum = 0;
	uint64_t primecount = 0;
	uint32_t factorcount = 0;
	uint64_t current_p;
	uint64_t last_P;
	uint64_t complete;
	uint64_t last_trickle = 0;
	int noprime = 0;
	time_t ckpt_curr, ckpt_last, boinc_last, boinc_curr;
	primesieve_iterator it;
	uint64_t my_P[8] __attribute__ ((aligned (64)));
	uint64_t resN[8] __attribute__ ((aligned (64)));
	__m512i vP, vPs, vK, vK1, x, KPOS, theN, theK;
	__mmask8 m, me, mt;
	__m512i v_checksum		= _mm512_setzero_si512();
	const __m512i ZERO		= _mm512_setzero_si512();
	const __m512i ONE		= _mm512_set1_epi64(1);
	const __m512i ST		= _mm512_set1_epi64(63);
	const __m512i VNSTEP		= _mm512_set1_epi64(data->sd.nstep);
	const __m512i Vr0		= _mm512_set1_epi64(data->sd.r0);
	const __m512i Vr1		= _mm512_set1_epi64(data->sd.r1);
	const uint32_t sm_mont_nstep	= data->sd.mont_nstep - 32;

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
	primesieve_jump_to(&it, current_p, data->pmax+12072);

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

	while(!done){

		for(int j=0; j<8; ++j){
			my_P[j] = primesieve_next_prime(&it);
			if(my_P[j] >= data->pmax){
				done = true;
				++noprime;
			}
		}

		vP = _mm512_load_epi64(my_P);

		// update BOINC fraction done every 5 sec
		time(&boinc_curr);
		if( ((int)boinc_curr - (int)boinc_last) > 4 ){
			boinc_last = boinc_curr;
			complete = my_P[0] - last_P;

			ckerr(pthread_mutex_lock(&lock2));
			g_complete += complete;
			if(data->id == 0){
				// update BOINC fraction done if thread 0
				double fd = (double)(g_complete)/data->sd.wu_range;
				if(boinc_is_standalone()) printf("Tests done: %.1f%%\n",fd*100.0);
				boinc_fraction_done(fd);
			}
			ckerr(pthread_mutex_unlock(&lock2));
	
			last_P = my_P[0];
		}

		// 1 minute checkpoint
		time(&ckpt_curr);
		if( ((int)ckpt_curr - (int)ckpt_last) > 60 ){
			ckpt_last = ckpt_curr;

			boinc_begin_critical_section();

			if(factorcount > 0){
				get_results_avx512(factorP, factorN, factorN, factorcount, data->id, data->sd.cw);
				factorcount = 0;
			}

			checksum += _mm512_reduce_add_epi64(v_checksum);
			v_checksum = _mm512_setzero_si512();

			checkpoint_thread( data->sd, my_P[0], data->id, state_a, primecount, checksum, last_trickle );

			boinc_end_critical_section();
		}

		vPs = neg_invmod2pow_ul_avx512(vP);

		vK = invpowmod_REDClr_avx512(vP, vPs, data->sd.nmin, Vr0, data->sd.bbits);


		uint32_t n = data->sd.nmin;

		do { 	
			// Remaining steps are all of equal size nstep
			// Select the even one.
			x	= _mm512_and_epi64(ONE, vK);
			me	= _mm512_cmpeq_epi64_mask(x, ONE);
			KPOS	= _mm512_mask_sub_epi64(vK, me, vP, vK);			// kpos = (((uint)k0) & 1)?(my_P - k0):k0

			x	= _mm512_sub_epi64(ZERO, KPOS);					// negate kpos
			x	= _mm512_and_epi64(KPOS, x);					// kpos & -kpos
			x	= _mm512_lzcnt_epi64(x);
			x	= _mm512_sub_epi64(ST, x);					// ctz(kpos)

			theK	= _mm512_srlv_epi64(KPOS, x);					// kpos >> tz
			theN	= _mm512_set1_epi64(n);
			theN	= _mm512_add_epi64(theN, x);					// n = n+tz

			mt	= _mm512_cmple_epi64_mask(x, VNSTEP);				// ctz(kpos) <= nstep?

			m	= _mm512_cmplt_epu64_mask(theK, theN);				// k < n?

			unsigned char allpass = _ktestz_mask8_u8(m, m);				// 1 if all k >= n

			while(!allpass){
				theN	= _mm512_mask_sub_epi64(theN, m, theN, ONE);		// n -= 1
				theK	= _mm512_mask_sllv_epi64(theK, m, theK, ONE);		// k <<= 1
				m	= _mm512_cmplt_epu64_mask(theK, theN);			// k < n?
				allpass = _ktestz_mask8_u8(m, m);				// 1 if all k >= n
			}

			m	= _mm512_cmpeq_epi64_mask(theN, theK);				// k == n?

			unsigned char nofactors = _ktestz_mask8_u8(m, mt);

			if(!nofactors){

				uint32_t evenmask = _cvtmask8_u32(me);
				uint32_t goodn = _cvtmask8_u32(_kand_mask8(m, mt));

				_mm512_store_epi64(resN, theN);

				for(int j=0; j<8; ++j){

					if( (goodn & 1) && my_P[j] < data->pmax ){

						int s = ( (evenmask&1) == 0 )?-1:1;
						uint64_t the_k = resN[j];
						uint64_t the_n = resN[j];

						if(the_n <= data->sd.nmax){
							if( goodfactor_cw(the_k, the_n, s)){
								if(try_all_factors(the_k, the_n, s) == 0){	// check for a small prime factor of the number
									uint64_t P = my_P[j];
									// check the factor actually divides the number
									if(verify_factor(P, the_k, the_n, s)){
//										printf("cw factor found, p %" PRIu64 " k %" PRIu64 " n %" PRIu64 " %d\n",P,the_k,the_n,s);
										factorP[factorcount] = (s==1) ? (int64_t)P : -((int64_t)P);
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
					evenmask >>= 1;
					goodn >>=1;
				}
			}


			// Proceed to the K for the next N.
			n += data->sd.nstep;

			if(data->sd.nstep < 32){
				vK = shiftmod_REDCsm_avx512 (vK, vP, _mm512_mul_epu32(vK, vPs), sm_mont_nstep, data->sd.nstep);
			}
			else if(data->sd.nstep == 32){
				vK = shiftmod_REDC32_avx512 (vK, vP, _mm512_mul_epu32(vK, vPs));
			}
			else{
				vK = shiftmod_REDC_avx512 (vK, vP, _mm512_mul_epu32(vK, vPs), data->sd.mont_nstep, data->sd.nstep);
			}


		} while (n < data->sd.nmax);

		// K and K1 should match if shiftmod loop calculated from nmin to nmax correctly
		vK1 = invpowmod_REDClr_avx512(vP, vPs, data->sd.lastN, Vr1, data->sd.bbits1);

		m = _mm512_cmpeq_epi64_mask(vK, vK1);

		unsigned char noerror = _kortestc_mask8_u8(m, m);

		if(!noerror){
			printf("ERROR: checksum calculation in thread %d\n",data->id);
			fprintf(stderr,"ERROR: checksum calculation in thread %d!\n",data->id);
			exit(EXIT_FAILURE);
		}

		// add P and K to checksum
		x		= _mm512_add_epi64(vP, vK);
		v_checksum	= _mm512_add_epi64(v_checksum, x);

		primecount += 8;

	}

	// adjust primecount and checksum for any P >= pmax, happens on last iteration
	primecount -= noprime;

	m = _mm512_cmpge_epi64_mask(vP, _mm512_set1_epi64(data->pmax));
	v_checksum = _mm512_mask_sub_epi64(v_checksum, m, v_checksum, vP);
	v_checksum = _mm512_mask_sub_epi64(v_checksum, m, v_checksum, vK);

	checksum += _mm512_reduce_add_epi64(v_checksum);

	// final checkpoint
	boinc_begin_critical_section();

	if(factorcount > 0){
		get_results_avx512(factorP, factorN, factorN, factorcount, data->id, data->sd.cw);
		factorcount = 0;
	}

	checkpoint_thread( data->sd, data->pmax, data->id, state_a, primecount, checksum, last_trickle );

	boinc_end_critical_section();

	// update global primecount and checksum
	ckerr(pthread_mutex_lock(&lock1));
	g_primecount += primecount;
	g_checksum += checksum;
	ckerr(pthread_mutex_unlock(&lock1));

	// update fraction done
	complete = data->pmax - last_P;

	ckerr(pthread_mutex_lock(&lock2));
	g_complete += complete;
	ckerr(pthread_mutex_unlock(&lock2));

	primesieve_free_iterator(&it);

	free(factorP);
	free(factorN);

	pthread_exit(NULL);

	return NULL;
}


void *thr_func_avx512(void *arg) {

	thread_data_t *data = (thread_data_t *)arg;
	bool state_a = true;
	bool done = false;
	uint64_t checksum = 0;
	uint64_t primecount = 0;
	uint32_t factorcount = 0;
	uint64_t current_p;
	uint64_t last_P;
	uint64_t complete;
	uint64_t last_trickle = 0;
	int noprime = 0;
	time_t ckpt_curr, ckpt_last, boinc_last, boinc_curr;
	primesieve_iterator it;
	uint64_t my_P[8] __attribute__ ((aligned (64)));
	uint64_t resN[8] __attribute__ ((aligned (64)));
	uint64_t resK[8] __attribute__ ((aligned (64)));
	__m512i vP, vPs, vK, vK1, x, KPOS, theN, theK;
	__mmask8 m, me, mt;
	__m512i v_checksum		= _mm512_setzero_si512();
	const __m512i ZERO		= _mm512_setzero_si512();
	const __m512i ONE		= _mm512_set1_epi64(1);
	const __m512i ST		= _mm512_set1_epi64(63);
	const __m512i VNSTEP		= _mm512_set1_epi64(data->sd.nstep);
	const __m512i Vr0		= _mm512_set1_epi64(data->sd.r0);
	const __m512i Vr1		= _mm512_set1_epi64(data->sd.r1);
	const __m512i KMAX		= _mm512_set1_epi64(data->sd.kmax);
	const __m512i KMIN		= _mm512_set1_epi64(data->sd.kmin);
	const uint32_t sm_mont_nstep	= data->sd.mont_nstep - 32;

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
	primesieve_jump_to(&it, current_p, data->pmax+12072);

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

	while(!done){

		for(int j=0; j<8; ++j){
			my_P[j] = primesieve_next_prime(&it);
			if(my_P[j] >= data->pmax){
				done = true;
				++noprime;
			}
		}

		vP = _mm512_load_epi64(my_P);

		// update BOINC fraction done every 5 sec
		time(&boinc_curr);
		if( ((int)boinc_curr - (int)boinc_last) > 4 ){
			boinc_last = boinc_curr;
			complete = my_P[0] - last_P;

			ckerr(pthread_mutex_lock(&lock2));
			g_complete += complete;
			if(data->id == 0){
				// update BOINC fraction done if thread 0
				double fd = (double)(g_complete)/data->sd.wu_range;
				if(boinc_is_standalone()) printf("Tests done: %.1f%%\n",fd*100.0);
				boinc_fraction_done(fd);
			}
			ckerr(pthread_mutex_unlock(&lock2));
	
			last_P = my_P[0];
		}

		// 1 minute checkpoint
		time(&ckpt_curr);
		if( ((int)ckpt_curr - (int)ckpt_last) > 60 ){
			ckpt_last = ckpt_curr;

			boinc_begin_critical_section();

			if(factorcount > 0){
				get_results_avx512(factorP, factorK, factorN, factorcount, data->id, data->sd.cw);
				factorcount = 0;
			}

			checksum += _mm512_reduce_add_epi64(v_checksum);
			v_checksum = _mm512_setzero_si512();

			checkpoint_thread( data->sd, my_P[0], data->id, state_a, primecount, checksum, last_trickle );

			boinc_end_critical_section();
		}

		vPs = neg_invmod2pow_ul_avx512(vP);

		vK = invpowmod_REDClr_avx512(vP, vPs, data->sd.nmin, Vr0, data->sd.bbits);


		uint32_t n = data->sd.nmin;

		do { 	
			// Remaining steps are all of equal size nstep
			// Select the even one.
			x	= _mm512_and_epi64(ONE, vK);
			me	= _mm512_cmpeq_epi64_mask(x, ONE);
			KPOS	= _mm512_mask_sub_epi64(vK, me, vP, vK);			// kpos = (((uint)k0) & 1)?(my_P - k0):k0

			x	= _mm512_sub_epi64(ZERO, KPOS);					// negate kpos
			x	= _mm512_and_epi64(KPOS, x);					// kpos & -kpos
			x	= _mm512_lzcnt_epi64(x);
			x	= _mm512_sub_epi64(ST, x);					// ctz(kpos)

			theK	= _mm512_srlv_epi64(KPOS, x);					// kpos >> tz
			theN	= _mm512_set1_epi64(n);
			theN	= _mm512_add_epi64(theN, x);					// n = n+tz

			mt	= _mm512_cmple_epi64_mask(x, VNSTEP);				// ctz(kpos) <= nstep?

			m	= _mm512_cmple_epi64_mask(theK, KMAX);				// k <= kmax?

			mt	= _kand_mask8(m, mt);

			m	= _mm512_cmpge_epi64_mask(theK, KMIN);				// k >= kmin?

			unsigned char nofactors = _ktestz_mask8_u8(m, mt);

			if(!nofactors){

				uint32_t evenmask = _cvtmask8_u32(me);
				uint32_t goodn = _cvtmask8_u32(_kand_mask8(m, mt));

				_mm512_store_epi64(resN, theN);
				_mm512_store_epi64(resK, theK);

				for(int j=0; j<8; ++j){

					if( (goodn & 1) && my_P[j] < data->pmax ){

						int s = ( (evenmask&1) == 0 )?-1:1;
						uint64_t the_k = resK[j];
						uint64_t the_n = resN[j];

						if(the_n <= data->sd.nmax){
							if( goodfactor(the_k, the_n, s)){
								if(try_all_factors(the_k, the_n, s) == 0){	// check for a small prime factor of the number
									uint64_t P = my_P[j];
									// check the factor actually divides the number
									if(verify_factor(P, the_k, the_n, s)){
//										printf("factor found, p %" PRIu64 " k %" PRIu64 " n %" PRIu64 " %d\n",P,the_k,the_n,s);
										factorP[factorcount] = (s==1) ? (int64_t)P : -((int64_t)P);
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
					evenmask >>= 1;
					goodn >>=1;
				}
			}


			// Proceed to the K for the next N.
			n += data->sd.nstep;

			if(data->sd.nstep < 32){
				vK = shiftmod_REDCsm_avx512 (vK, vP, _mm512_mul_epu32(vK, vPs), sm_mont_nstep, data->sd.nstep);
			}
			else if(data->sd.nstep == 32){
				vK = shiftmod_REDC32_avx512 (vK, vP, _mm512_mul_epu32(vK, vPs));
			}
			else{
				vK = shiftmod_REDC_avx512 (vK, vP, _mm512_mul_epu32(vK, vPs), data->sd.mont_nstep, data->sd.nstep);
			}


		} while (n < data->sd.nmax);

		// K and K1 should match if shiftmod loop calculated from nmin to nmax correctly
		vK1 = invpowmod_REDClr_avx512(vP, vPs, data->sd.lastN, Vr1, data->sd.bbits1);

		m = _mm512_cmpeq_epi64_mask(vK, vK1);

		unsigned char noerror = _kortestc_mask8_u8(m, m);

		if(!noerror){
			printf("ERROR: checksum calculation in thread %d\n",data->id);
			fprintf(stderr,"ERROR: checksum calculation in thread %d!\n",data->id);
			exit(EXIT_FAILURE);
		}

		// add P and K to checksum
		x		= _mm512_add_epi64(vP, vK);
		v_checksum	= _mm512_add_epi64(v_checksum, x);

		primecount += 8;

	}

	// adjust primecount and checksum for any P >= pmax, happens on last iteration
	primecount -= noprime;

	m = _mm512_cmpge_epi64_mask(vP, _mm512_set1_epi64(data->pmax));
	v_checksum = _mm512_mask_sub_epi64(v_checksum, m, v_checksum, vP);
	v_checksum = _mm512_mask_sub_epi64(v_checksum, m, v_checksum, vK);

	checksum += _mm512_reduce_add_epi64(v_checksum);

	// final checkpoint
	boinc_begin_critical_section();

	if(factorcount > 0){
		get_results_avx512(factorP, factorK, factorN, factorcount, data->id, data->sd.cw);
		factorcount = 0;
	}

	checkpoint_thread( data->sd, data->pmax, data->id, state_a, primecount, checksum, last_trickle );

	boinc_end_critical_section();

	// update global primecount and checksum
	ckerr(pthread_mutex_lock(&lock1));
	g_primecount += primecount;
	g_checksum += checksum;
	ckerr(pthread_mutex_unlock(&lock1));

	// update fraction done
	complete = data->pmax - last_P;

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


