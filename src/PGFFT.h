#ifndef PGFFT_H
#define PGFFT_H

#include <vector>
#include <complex>




class PGFFT {
public:

   // initialize data strctures for n-point FFT
   // REQUIREMENT: n > 0
   explicit PGFFT(long n);

   // apply n-point FFT to v
   // REQUIREMENT: v.size() == n
   void apply(std::vector<std::complex<double>>& v) const;


private:
   long n;
   long k;

   long strategy;

   // holds all of the twiddle factors
   std::vector<std::vector<std::complex<double>>> tab;

   // additional data structures needed for Bluestein
   std::vector<std::complex<double>> powers;
   std::vector<std::complex<double>> Rb;

   // additonal data structures needed for 2^k-point FFT
   std::vector<long> rev, rev1;


};


#endif
