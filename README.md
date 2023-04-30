# PCWSieveCPU

PCWSieveCPU by Bryan Little

A BOINC-enabled stand-alone sieve for Proth (k·2n+1), Cullen (n·2n+1), and Woodall (n·2n−1) factors.
Multithreading and AVX512 are supported.

+1-1 search algorithm by
* Geoffrey Reynolds
* Ken Brazier, 2009

With contributions by
* Yves Gallot
* Kim Walisch

## Requirements

* 64 bit operating system.

## Running the program
```
command line options
* -p #
* -P #			Sieve primes -p <= p < -P < 2^62
* -k #
* -K #			Sieve for primes k*2^n+/-1 with -k <= k <= -K < 2^32
* -n #
* -N #			Sieve for primes k*2^n+/-1 with 65 <= -n <= n <= -N < 2^32
* -c			Search for Cullen/Woodall factors
* -s or --test		Perform self test to verify proper operation of the program.
* -z or --noavx512	Disable AVX512.
* -t # or --nthreads #	Multithreading, use # threads.
```
## Related Links

* [PSieve-CUDA](https://github.com/Ken-g6/PSieve-CUDA)
* [primesieve](https://github.com/kimwalisch/primesieve)
