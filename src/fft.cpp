/* Copyright (C) 2012-2017 IBM Corp.
 * This program is Licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */
/**
 * @file fft.cpp - computing the canonical embedding and related norms
 **/
#include <complex>
#include <cmath>
#include <numeric> // std::accumulate
#include <algorithm>
#include <NTL/BasicThreadPool.h>
#include "NumbTh.h"
#include "timing.h"
#include "norms.h"
#include "PAlgebra.h"
NTL_CLIENT


//=========================================================

/* 
 * Free FFT and convolution (C++)
 * 
 * Copyright (c) 2017 Project Nayuki. (MIT License)
 * https://www.nayuki.io/page/free-small-fft-in-multiple-languages
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * - The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 * - The Software is provided "as is", without warranty of any kind, express or
 *   implied, including but not limited to the warranties of merchantability,
 *   fitness for a particular purpose and noninfringement. In no event shall the
 *   authors or copyright holders be liable for any claim, damages or other
 *   liability, whether in an action of contract, tort or otherwise, arising from,
 *   out of or in connection with the Software or the use or other dealings in the
 *   Software.
 */

namespace Fft {
   
   /* 
    * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
    * The vector can have any length. This is a wrapper function.
    */
   void transform(std::vector<std::complex<double> > &vec);
   
   
   /* 
    * Computes the inverse discrete Fourier transform (IDFT) of the given complex vector, storing the result back into the vector.
    * The vector can have any length. This is a wrapper function. This transform does not perform scaling, so the inverse is not a true inverse.
    */
   void inverseTransform(std::vector<std::complex<double> > &vec);
   
   
   /* 
    * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
    * The vector's length must be a power of 2. Uses the Cooley-Tukey decimation-in-time radix-2 algorithm.
    */
   void transformRadix2(std::vector<std::complex<double> > &vec);
   
   
   /* 
    * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
    * The vector can have any length. This requires the convolution function, which in turn requires the radix-2 FFT function.
    * Uses Bluestein's chirp z-transform algorithm.
    */
   void transformBluestein(std::vector<std::complex<double> > &vec);
   
   
   /* 
    * Computes the circular convolution of the given complex vectors. Each vector's length must be the same.
    */
   void convolve(
      const std::vector<std::complex<double> > &vecx,
      const std::vector<std::complex<double> > &vecy,
      std::vector<std::complex<double> > &vecout);
   
}




// Private function prototypes
static size_t reverseBits(size_t x, int n);


void Fft::transform(vector<complex<double> > &vec) {
   size_t n = vec.size();
   if (n == 0)
      return;
   else if ((n & (n - 1)) == 0)  // Is power of 2
      transformRadix2(vec);
   else  // More complicated algorithm for arbitrary sizes
      transformBluestein(vec);
}


void Fft::inverseTransform(vector<complex<double> > &vec) {
   std::transform(vec.cbegin(), vec.cend(), vec.begin(),
      static_cast<complex<double> (*)(const complex<double> &)>(std::conj));
   transform(vec);
   std::transform(vec.cbegin(), vec.cend(), vec.begin(),
      static_cast<complex<double> (*)(const complex<double> &)>(std::conj));
}


void Fft::transformRadix2(vector<complex<double> > &vec) {
   FHE_NTIMER_START(AAA_transformRadix2);
   // Length variables
   size_t n = vec.size();
   int levels = 0;  // Compute levels = floor(log2(n))
   for (size_t temp = n; temp > 1U; temp >>= 1)
      levels++;
   if (static_cast<size_t>(1U) << levels != n)
      throw std::domain_error("Length is not a power of 2");

   FHE_NTIMER_START(AAA_table1);
   
   // Trignometric table
   vector<complex<double> > expTable(n / 2);
   for (size_t i = 0; i < n / 2; i++)
      expTable[i] = std::exp(complex<double>(0, -2 * M_PI * i / n));

   FHE_NTIMER_STOP(AAA_table1);

   FHE_NTIMER_START(AAA_reverseBits);
   
   // Bit-reversed addressing permutation
   for (size_t i = 0; i < n; i++) {
      size_t j = reverseBits(i, levels);
      if (j > i)
         std::swap(vec[i], vec[j]);
   }

   FHE_NTIMER_STOP(AAA_reverseBits);
   
   FHE_NTIMER_START(AAA_cooley_tukey);
   // Cooley-Tukey decimation-in-time radix-2 FFT
   for (size_t size = 2; size <= n; size *= 2) {
      size_t halfsize = size / 2;
      size_t tablestep = n / size;
      for (size_t i = 0; i < n; i += size) {
         for (size_t j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
            complex<double> temp = vec[j + halfsize] * expTable[k];
            vec[j + halfsize] = vec[j] - temp;
            vec[j] += temp;
         }
      }
      if (size == n)  // Prevent overflow in 'size *= 2'
         break;
   }
}


void Fft::transformBluestein(vector<complex<double> > &vec) {
   // Find a power-of-2 convolution length m such that m >= n * 2 + 1
   size_t n = vec.size();
   size_t m = 1;
   while (m / 2 <= n) {
      if (m > SIZE_MAX / 2)
         throw std::length_error("Vector too large");
      m *= 2;
   }

   FHE_NTIMER_START(AAA_table2);
   
   // Trignometric table
   vector<complex<double> > expTable(n);
   for (size_t i = 0; i < n; i++) {
      unsigned long long temp = static_cast<unsigned long long>(i) * i;
      temp %= static_cast<unsigned long long>(n) * 2;
      double angle = M_PI * temp / n;
      // Less accurate alternative if long long is unavailable: double angle = M_PI * i * i / n;
      expTable[i] = std::exp(complex<double>(0, -angle));
   }

   FHE_NTIMER_STOP(AAA_table2);
   
   // Temporary vectors and preprocessing
   vector<complex<double> > av(m);
   for (size_t i = 0; i < n; i++)
      av[i] = vec[i] * expTable[i];
   vector<complex<double> > bv(m);
   bv[0] = expTable[0];
   for (size_t i = 1; i < n; i++)
      bv[i] = bv[m - i] = std::conj(expTable[i]);
   
   // Convolution
   vector<complex<double> > cv(m);
   convolve(av, bv, cv);
   
   // Postprocessing
   for (size_t i = 0; i < n; i++)
      vec[i] = cv[i] * expTable[i];
}


void Fft::convolve(
      const vector<complex<double> > &xvec,
      const vector<complex<double> > &yvec,
      vector<complex<double> > &outvec) {

   FHE_NTIMER_START(AAA_convolve);
   
   size_t n = xvec.size();
   if (n != yvec.size() || n != outvec.size())
      throw std::domain_error("Mismatched lengths");
   vector<complex<double> > xv = xvec;
   vector<complex<double> > yv = yvec;
   transform(xv);
   transform(yv);
   for (size_t i = 0; i < n; i++)
      xv[i] *= yv[i];
   inverseTransform(xv);
   for (size_t i = 0; i < n; i++)  // Scaling (because this FFT implementation omits it)
      outvec[i] = xv[i] / static_cast<double>(n);
}


static size_t reverseBits(size_t x, int n) {
   size_t result = 0;
   for (int i = 0; i < n; i++, x >>= 1)
      result = (result << 1) | (x & 1U);
   return result;
}


//=========================================================






const double pi = 4 * std::atan(1);

#ifdef FFT_ARMA
#warning "canonicalEmbedding implemented via Armadillo"
#include <armadillo>
void convert(zzX& to, const arma::vec& from)
{
  to.SetLength(from.size());
  NTL_EXEC_RANGE(to.length(), first, last)
  for (long i=first; i<last; i++)
    to[i] = std::round(from[i]);
  NTL_EXEC_RANGE_END
}

void convert(arma::vec& to, const zzX& from)
{
  to.resize(from.length());
  NTL_EXEC_RANGE(from.length(), first, last)
  for (long i=first; i<last; i++)
    to[i] = from[i];
  NTL_EXEC_RANGE_END
}

void convert(arma::vec& to, const ZZX& from)
{
  to.resize(from.rep.length());
  NTL_EXEC_RANGE(from.rep.length(), first, last)
  for (long i=first; i<last; i++) {
    double x = conv<double>(from[i]);
    to[i] = x;
  }
  NTL_EXEC_RANGE_END
}

void convert(arma::vec& to, const Vec<double>& from)
{
  to.resize(from.length());
  for (long i: range(from.length()))
    to[i] = from[i];
}

#if 0

//======================

namespace arma {

  template<> 
  struct is_supported_elem_type<RR> {
    enum {value = 1};
  }; 

  template<> 
  struct is_supported_elem_type<cx_RR> {
    enum {value = 1};
  }; 

  template<> 
  struct is_real<RR> {
    enum {value = 1};
  }; 

}


void convert(zzX& to, const arma::Col<RR>& from)
{
  to.SetLength(from.size());
  for (long i: range(to.length()))
    to[i] = conv<long>(RoundToZZ(from[i]));
}

void convert(arma::Col<RR>& to, const zzX& from)
{
  to.resize(from.length());
  for (long i: range(from.length()))
    to[i] = from[i];
}

void convert(arma::Col<RR>& to, const ZZX& from)
{
  to.resize(from.rep.length());
  for (long i: range(from.rep.length())) {
    RR x = conv<RR>(from[i]);
    to[i] = x;
  }
}

#endif

// Computing the canonical embedding. This function returns in v only
// the first half of the entries, the others are v[phi(m)-i]=conj(v[i])
void canonicalEmbedding(std::vector<cx_double>& v,
                        const zzX& f, const PAlgebra& palg)
{
  FHE_TIMER_START;
  long m = palg.getM();
  long phimBy2 = divc(palg.getPhiM(),2);
  arma::vec av; // convert to vector of doubles
  convert(av, f);
  arma::cx_vec avv = arma::fft(av,m); // compute the full FFT

  v.resize(phimBy2); // the first half of Zm*

  if (palg.getNSlots()==phimBy2) // order roots by the palg order
    for (long i=0; i<phimBy2; i++)
      v[phimBy2-i-1] = avv[palg.ith_rep(i)];
  else                           // order roots sequentially
    for (long i=1, idx=0; i<=m/2; i++)
      if (palg.inZmStar(i)) v[idx++] = avv[i];
}

void canonicalEmbedding(std::vector<cx_double>& v,
                        const ZZX& f, const PAlgebra& palg)
{
  FHE_TIMER_START;
  long m = palg.getM();
  long phimBy2 = divc(palg.getPhiM(),2);
  arma::vec av; // convert to vector of doubles
  convert(av, f);

  arma::cx_vec avv = arma::fft(av,m); // compute the full FFT

  v.resize(phimBy2); // the first half of Zm*

  if (palg.getNSlots()==phimBy2) // order roots by the palg order
    for (long i=0; i<phimBy2; i++)
      v[phimBy2-i-1] = avv[palg.ith_rep(i)];
  else                           // order roots sequentially
    for (long i=1, idx=0; i<=m/2; i++)
      if (palg.inZmStar(i)) v[idx++] = avv[i];
}

#if 0
void canonicalEmbedding(std::vector<cx_double>& v,
                        const Vec<double>& f, const PAlgebra& palg)
{
  FHE_TIMER_START;
  long m = palg.getM();
  long phimBy2 = divc(palg.getPhiM(),2);
  arma::vec av; // convert to vector of doubles
  convert(av, f);

  FHE_NTIMER_START(AAA_arma_fft);
  arma::cx_vec avv = arma::fft(av,m); // compute the full FFT
  FHE_NTIMER_STOP(AAA_arma_fft);

  v.resize(phimBy2); // the first half of Zm*

  if (palg.getNSlots()==phimBy2) // order roots by the palg order
    for (long i=0; i<phimBy2; i++)
      v[phimBy2-i-1] = avv[palg.ith_rep(i)];
  else                           // order roots sequentially
    for (long i=1, idx=0; i<=m/2; i++)
      if (palg.inZmStar(i)) v[idx++] = avv[i];
}
#else
void canonicalEmbedding(std::vector<cx_double>& v,
                        const Vec<double>& f, const PAlgebra& palg)
{
  FHE_TIMER_START;
  long m = palg.getM();
  long phimBy2 = divc(palg.getPhiM(),2);
  vector<complex<double>> w(m);
  for (long i: range(f.length())) 
    w[i] = f[i];
  for (long i: range(f.length(), m))
    w[i] = 0;

  FHE_NTIMER_START(AAA_arma_fft);
  Fft::transform(w);
  FHE_NTIMER_STOP(AAA_arma_fft);
  

  v.resize(phimBy2); // the first half of Zm*

  if (palg.getNSlots()==phimBy2) // order roots by the palg order
    for (long i=0; i<phimBy2; i++)
      v[phimBy2-i-1] = w[palg.ith_rep(i)];
  else                           // order roots sequentially
    for (long i=1, idx=0; i<=m/2; i++)
      if (palg.inZmStar(i)) v[idx++] = w[i];
}
#endif


#if 0
void canonicalEmbedding(std::vector<cx_RR>& v,
                        const ZZX& f, const PAlgebra& palg)
{
  FHE_TIMER_START;
  long m = palg.getM();
  long phimBy2 = divc(palg.getPhiM(),2);
  arma::Col<RR> av; // convert to vector of doubles
  convert(av, f);
  arma::Col<cx_RR> avv = arma::fft(av,m); // compute the full FFT

  v.resize(phimBy2); // the first half of Zm*

  if (palg.getNSlots()==phimBy2) // order roots by the palg order
    for (long i=0; i<phimBy2; i++)
      v[phimBy2-i-1] = avv[palg.ith_rep(i)];
  else                           // order roots sequentially
    for (long i=1, idx=0; i<=m/2; i++)
      if (palg.inZmStar(i)) v[idx++] = avv[i];
}
#endif

// Roughly the inverse of canonicalEmbedding, except for scaling and
// rounding issues. Calling embedInSlots(f,v,palg,1.0,strictInverse=true)
// after setting canonicalEmbedding(v, f, palg), is sure to recover the
// same f, but embedInSlots(f,v,palg,1.0,strictInverse=false) may return
// a different "nearby" f.
void embedInSlots(zzX& f, const std::vector<cx_double>& v,
                  const PAlgebra& palg, double scaling, bool strictInverse)
{
  FHE_TIMER_START;
  long m = palg.getM();
  long phimBy2 = divc(palg.getPhiM(),2);
  arma::cx_vec avv(m);
  for (auto& x: avv) x = 0.0;

  if (palg.getNSlots()==phimBy2) // roots ordered by the palg order
    for (long i=0; i<palg.getNSlots(); i++) {
      long j = palg.ith_rep(i);
      long ii = palg.getNSlots()-i-1;
      if (ii < lsize(v)) {
        avv[j] = scaling*v[ii];
        avv[m-j] = std::conj(avv[j]);
      }
    }
  else                           // roots ordered sequentially
    for (long i=1, idx=0; i<=m/2 && idx<lsize(v); i++) {
      if (palg.inZmStar(i)) {
        avv[i] = scaling*v[idx++];
        avv[m-i] = std::conj(avv[i]);
      }
    }
  arma::vec av = arma::real(arma::ifft(avv,m)); // compute the inverse FFT

  // If v was obtained by canonicalEmbedding(v,f,palg,1.0) then we have
  // the guarantee that m*av is an integral polynomial, and moreover
  // m*av mod Phi_m(x) is in m*Z[X].
  if (strictInverse) av *= m; // scale up by m
  convert(f, av);    // round to an integer polynomial
  reduceModPhimX(f, palg);
  if (strictInverse) f /= m;  // scale down by m
  normalize(f);
}
#else
#ifdef FFT_NATIVE
#warning "canonicalEmbedding implemented via slow DFT, expect very slow key-generation"
// An extremely lame implementation of the canonical embedding

// evaluate poly(x) using Horner's rule
cx_double complexEvalPoly(const zzX& poly, const cx_double& x)
{
  if (lsize(poly)<=0) return cx_double(0.0,0.0);
  cx_double res(double(poly[0]), 0.0);
  for (long i=1; i<lsize(poly); i++) {
    res *= x;
    res += cx_double(double(poly[i]));
  }
  return res;
}

void canonicalEmbedding(std::vector<cx_double>& v, const zzX& f, const PAlgebra& palg)
{
  FHE_TIMER_START;
  long m = palg.getM();
  long phimBy2 = divc(palg.getPhiM(),2);
  vector<long> zmstar(phimBy2); // the first half of Zm*

  if (palg.getNSlots()==phimBy2) // order roots by the palg order
    for (long i=0; i<phimBy2; i++)
      zmstar[phimBy2-i-1] = palg.ith_rep(i);
  else                           // order roots sequentially
    for (long i=1, idx=0; i<=m/2; i++)
      if (palg.inZmStar(i)) zmstar[idx++] = i;

  v.resize(phimBy2);
  NTL_EXEC_RANGE(phimBy2, first, last)
  for (long i=first; i < last; ++i) {
    auto rou = std::polar<double>(1.0, -(2*pi*zmstar[i])/m); // root of unity
    v[i] = complexEvalPoly(f,rou);
  }
  NTL_EXEC_RANGE_END
  FHE_TIMER_STOP;
}

// evaluate poly(x) using Horner's rule
// FIXME: this is actually evaluating the reverse polynomial
cx_double complexEvalPoly(const Vec<double>& poly, const cx_double& x)
{
  if (poly.length()<=0) return cx_double(0.0,0.0);
  cx_double res(poly[0], 0.0);
  for (long i: range(1, poly.length())) {
    res *= x;
    res += cx_double(poly[i]);
  }
  return res;
}

void canonicalEmbedding(std::vector<cx_double>& v, const ZZX& f, const PAlgebra& palg)
{
  FHE_TIMER_START;
  long m = palg.getM();
  long phimBy2 = divc(palg.getPhiM(),2);
  vector<long> zmstar(phimBy2); // the first half of Zm*

  Vec<double> ff;
  conv(ff, f.rep);

  if (palg.getNSlots()==phimBy2) // order roots by the palg order
    for (long i=0; i<phimBy2; i++)
      zmstar[phimBy2-i-1] = palg.ith_rep(i);
  else                           // order roots sequentially
    for (long i=1, idx=0; i<=m/2; i++)
      if (palg.inZmStar(i)) zmstar[idx++] = i;

  v.resize(phimBy2);
  NTL_EXEC_RANGE(phimBy2, first, last)
  for (long i=first; i < last; ++i) {
    auto rou = std::polar<double>(1.0, -(2*pi*zmstar[i])/m); // root of unity
    v[i] = complexEvalPoly(ff,rou);
  }
  NTL_EXEC_RANGE_END
  FHE_TIMER_STOP;
}

void embedInSlots(zzX& f, const std::vector<cx_double>& v,
                  const PAlgebra& palg, double scaling, bool strictInverse)
{
  throw helib::LogicError("embedInSlots not implemented with FFT_NATIVE");
}
#endif // ifdef FFT_NATIVE
#endif // ifdef FFT_ARMA
