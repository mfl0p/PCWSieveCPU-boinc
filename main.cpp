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
#include <getopt.h>

#include "boinc_api.h"
#include "boinc_opencl.h"
#include "primesieve.h"
#include "putil.h"
#include "cpu_sieve.h"

using namespace std; 


void help()
{
	printf("Program usage\n");
	printf("-p #\n");
	printf("-P #			Sieve primes -p <= p < -P < 2^62\n");
	printf("-k #\n");
	printf("-K #			Sieve for primes k*2^n+/-1 with -k <= k <= -K < 2^32\n");
	printf("-n #\n");
	printf("-N # 			Sieve for primes k*2^n+/-1 with 65 <= -n <= n <= -N < 2^32\n");
	printf("-c 			Search for Cullen/Woodall factors\n");
	printf("-s or --test		Perform self test to verify proper operation of the program.\n");
	printf("-z or --noavx512	Disable AVX512.\n");
	printf("-t # or --nthreads #	Multithreading, use # threads.\n");
	printf("-h			Print this help\n");
        boinc_finish(EXIT_FAILURE);
}


static const char *short_opts = "p:P:k:K:n:N:cszt:h";

static int parse_option(int opt, char *arg, const char *source, searchData & sd)
{
  int status = 0;

  switch (opt)
  {
    case 'p':
      status = parse_uint64(&sd.pmin,arg,3,(UINT64_C(1)<<62)-1);
      break;

    case 'P':
      status = parse_uint64(&sd.pmax,arg,4,(UINT64_C(1)<<62)-1);
      break;

    case 'k':
      status = parse_uint(&sd.kmin,arg,1,(1U<<31)-1);
      break;

    case 'K':
      status = parse_uint(&sd.kmax,arg,1,(1U<<31)-1);
      break;
      
    case 'n':
      status = parse_uint(&sd.nmin,arg,65,(1U<<31)-1);
      break;

    case 'N':
      status = parse_uint(&sd.nmax,arg,65,(1U<<31)-1);
      break;

    case 'c':
      sd.cw = true;
      fprintf(stderr,"Searching for Cullen/Woodall factors.\n");
      printf("Searching for Cullen/Woodall factors.\n");
      break;

    case 's':
      sd.test = true;
      fprintf(stderr,"Performing self test.\n");
      printf("Performing self test.\n");
      break;

    case 'z':
      sd.avx512 = false;
      fprintf(stderr,"Disabled AVX512.\n");
      printf("Disabled AVX512.\n");
      break;

    case 't':
      status = parse_uint(&sd.threads,arg,1,(1U<<31)-1);
      fprintf(stderr,"Using %u threads.\n", sd.threads);
      printf("Using %u threads.\n", sd.threads);
      break;

    case 'h':
      help();
      break;

    case '?':
      status = -3;
      break;
  }

  return status;
}

static const struct option long_opts[] = {
  {"noavx512",  no_argument, 0, 'z'},
  {"test",  no_argument, 0, 's'},
  {"nthreads",  required_argument, 0, 't'},
  {0,0,0,0}
};


/* Process command-line options using getopt_long().
   Non-option arguments are treated as if they belong to option zero.
   Returns the number of options processed.
 */
static int process_args(int argc, char *argv[], searchData & sd)
{
  int count = 0, ind = -1, opt;

  while ((opt = getopt_long(argc,argv,short_opts,long_opts,&ind)) != -1)
    switch (parse_option(opt,optarg,NULL,sd))
    {
      case 0:
        ind = -1;
        count++;
        break;

      case -1:
        /* If ind is unchanged then this is a short option, otherwise long. */
        if (ind == -1){
          printf("%s: invalid argument -%c %s\n",argv[0],opt,optarg);
          fprintf(stderr,"%s: invalid argument -%c %s\n",argv[0],opt,optarg);
	}
        else{
     	  printf("%s: invalid argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
          fprintf(stderr,"%s: invalid argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
	}
        boinc_finish(EXIT_FAILURE);

      case -2:
        /* If ind is unchanged then this is a short option, otherwise long. */
        if (ind == -1){
          printf("%s: out of range argument -%c %s\n",argv[0],opt,optarg);
          fprintf(stderr,"%s: out of range argument -%c %s\n",argv[0],opt,optarg);
	}
        else{
          printf("%s: out of range argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
          fprintf(stderr,"%s: out of range argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
	}
        boinc_finish(EXIT_FAILURE);

      default:
        printf("unknown command line argument\n");
        boinc_finish(EXIT_FAILURE);
    }

  while (optind < argc)
    switch (parse_option(0,argv[optind],NULL,sd))
    {
      case 0:
        optind++;
        count++;
        break;

      case -1:
        fprintf(stderr,"%s: invalid non-option argument %s\n",argv[0],argv[optind]);
        boinc_finish(EXIT_FAILURE);

      case -2:
        fprintf(stderr,"%s: out of range non-option argument %s\n",argv[0],argv[optind]);
        boinc_finish(EXIT_FAILURE);

      default:
        boinc_finish(EXIT_FAILURE);
    }


  return count;
}





int main(int argc, char *argv[])
{ 
	searchData sd;

	primesieve_set_num_threads(1);

	__builtin_cpu_init();
	if( __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512cd") ) sd.avx512 = true;

        // Initialize BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
	options.multi_thread = true; 
        boinc_init_options(&options);

	fprintf(stderr, "\nPCWSieve version %s by Bryan Little, Ken Brazier, Geoffrey Reynolds\n",VERS);
	fprintf(stderr, "Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	if(boinc_is_standalone()){
		printf("PCWSieve version %s by Bryan Little, Ken Brazier, Geoffrey Reynolds\n",VERS);
		printf("Compiled " __DATE__ " with GCC " __VERSION__ "\n");

	}

        // Print out cmd line for diagnostics
        fprintf(stderr, "Command line: ");
        for (int i = 0; i < argc; i++)
        	fprintf(stderr, "%s ", argv[i]);
        fprintf(stderr, "\n");

	process_args(argc,argv,sd);

	if(sd.avx512){
		printf("Using AVX512.\n");
		fprintf(stderr, "Using AVX512.\n");
	}

	if(sd.test == true){
		run_test(sd);

	}
	else{
		cpu_sieve(sd);
	}


	boinc_finish(EXIT_SUCCESS);

	return 0; 
} 

