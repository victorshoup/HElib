#include <NTL/ZZ.h>

#include <iostream>


#include <vector>
#include <complex>


class PGFFT {
   long n;
   long k;

   std::vector<std::vector<std::complex<double>>> tab;
   std::vector<long> rev;



};




using std::vector;
using std::complex;

typedef complex<double> cmplx_t;


#if (defined(__GNUC__) && (__GNUC__ >= 4))

// on relative modern versions of gcc, we can
// decalare "restricted" pointers in C++

#define PGFFT_RESTRICT __restrict

#else

#define PGFFT_RESTRICT

#endif





#define fwd_butterfly(xx0, xx1, w)  \
do \
{ \
   cmplx_t x0_ = xx0; \
   cmplx_t x1_ = xx1; \
   cmplx_t t_  = x0_ -  x1_; \
   xx0 = x0_ + x1_; \
   xx1 = t_ * w; \
}  \
while (0)



#define fwd_butterfly0(xx0, xx1) \
do   \
{  \
   cmplx_t x0_ = xx0;  \
   cmplx_t x1_ = xx1;  \
   xx0 = x0_ + x1_; \
   xx1 = x0_ - x1_; \
}  \
while (0)




// requires size divisible by 8
static void
new_fft_layer(cmplx_t* xp, long blocks, long size,
              const cmplx_t* PGFFT_RESTRICT wtab)
{
  size /= 2;

  do
    {
      cmplx_t* PGFFT_RESTRICT xp0 = xp;
      cmplx_t* PGFFT_RESTRICT xp1 = xp + size;

      // first 4 butterflies
      fwd_butterfly0(xp0[0+0], xp1[0+0]);
      fwd_butterfly(xp0[0+1], xp1[0+1], wtab[0+1]);
      fwd_butterfly(xp0[0+2], xp1[0+2], wtab[0+2]);
      fwd_butterfly(xp0[0+3], xp1[0+3], wtab[0+3]);

      // 4-way unroll
      for (long j = 4; j < size; j += 4) {
        fwd_butterfly(xp0[j+0], xp1[j+0], wtab[j+0]);
        fwd_butterfly(xp0[j+1], xp1[j+1], wtab[j+1]);
        fwd_butterfly(xp0[j+2], xp1[j+2], wtab[j+2]);
        fwd_butterfly(xp0[j+3], xp1[j+3], wtab[j+3]);
      }

      xp += 2 * size;
    }
  while (--blocks != 0);
}



static void
new_fft_last_two_layers(cmplx_t* xp, long blocks, const cmplx_t* wtab)
{
  // 4th root of unity
  cmplx_t w = wtab[1];

  do
    {
      cmplx_t u0 = xp[0];
      cmplx_t u1 = xp[1];
      cmplx_t u2 = xp[2];
      cmplx_t u3 = xp[3];

      cmplx_t v0 = u0 + u2;
      cmplx_t v2 = u0 - u2;
      cmplx_t v1 = u1 + u3; 
      cmplx_t t  = u1 - u3; 
      cmplx_t v3 = t * w;

      xp[0] = v0 + v1;
      xp[1] = v0 - v1;
      xp[2] = v2 + v3;
      xp[3] = v2 - v3; 

      xp += 4;
    }
  while (--blocks != 0);
}


static void 
new_fft_base(cmplx_t* xp, long lgN, const vector<vector<cmplx_t>>& tab)
{
  if (lgN == 0) return;

  if (lgN == 1)
    {
      cmplx_t x0 = xp[0];
      cmplx_t x1 = xp[1];
      xp[0] = x0 + x1;
      xp[1] = x0 - x1;
      return;
    }


  long N = 1L << lgN;

  for (long j = lgN, size = N, blocks = 1;
       j > 2; j--, blocks <<= 1, size >>= 1)
    new_fft_layer(xp, blocks, size, &tab[j][0]);

  new_fft_last_two_layers(xp, N/4, &tab[2][0]);
}


// Implements the truncated FFT interface, described above.
// All computations done in place, and xp should point to 
// an array of size N, all of which may be overwitten
// during the computation.

#define PGFFT_NEW_FFT_THRESH (12)

static
void new_fft_short(cmplx_t* xp, long yn, long xn, long lgN, 
                   const vector<vector<cmplx_t>>& tab)
{
  long N = 1L << lgN;

  if (yn == N)
    {
      if (xn == N && lgN <= PGFFT_NEW_FFT_THRESH)
	{
	  // no truncation
	  new_fft_base(xp, lgN, tab);
	  return;
	}
    }

  // divide-and-conquer algorithm

  long half = N >> 1;

  if (yn <= half)
    {
      if (xn <= half)
	{
	  new_fft_short(xp, yn, xn, lgN - 1, tab);
	}
      else
	{
	  xn -= half;

	  // (X, Y) -> X + Y
	  for (long j = 0; j < xn; j++)
	    xp[j] = xp[j] + xp[j + half];

	  new_fft_short(xp, yn, half, lgN - 1, tab);
	}
    }
  else
    {
      yn -= half;
      
      cmplx_t* PGFFT_RESTRICT xp0 = xp;
      cmplx_t* PGFFT_RESTRICT xp1 = xp + half;
      const cmplx_t* PGFFT_RESTRICT wtab = &tab[lgN][0];

      if (xn <= half)
	{
	  // X -> (X, w*X)
	  for (long j = 0; j < xn; j++)
	    xp1[j] = xp0[j] * wtab[j];

	  new_fft_short(xp0, half, xn, lgN - 1, tab);
	  new_fft_short(xp1, yn, xn, lgN - 1, tab);
	}
      else
	{
	  xn -= half;

	  // (X, Y) -> (X + Y, w*(X - Y))
          // DIRT: assumes xn is a multiple of 4
          fwd_butterfly0(xp0[0], xp1[0]);
          fwd_butterfly(xp0[1], xp1[1], wtab[1]);
          fwd_butterfly(xp0[2], xp1[2], wtab[2]);
          fwd_butterfly(xp0[3], xp1[3], wtab[3]);
	  for (long j = 4; j < xn; j+=4) {
            fwd_butterfly(xp0[j+0], xp1[j+0], wtab[j+0]);
            fwd_butterfly(xp0[j+1], xp1[j+1], wtab[j+1]);
            fwd_butterfly(xp0[j+2], xp1[j+2], wtab[j+2]);
            fwd_butterfly(xp0[j+3], xp1[j+3], wtab[j+3]);
          }

	  // X -> (X, w*X)
	  for (long j = xn; j < half; j++)
	    xp1[j] = xp0[j] * wtab[j];

	  new_fft_short(xp0, half, half, lgN - 1, tab);
	  new_fft_short(xp1, yn, half, lgN - 1, tab);
	}
    }
}

static void new_fft(cmplx_t* xp, long lgN, const vector<vector<cmplx_t>>& tab)
{
   long N = 1L << lgN;
   new_fft_short(xp, N, N, lgN, tab);
}





#define inv_butterfly0(xx0, xx1)  \
do   \
{  \
   cmplx_t x0_ = xx0;  \
   cmplx_t x1_ = xx1;  \
   xx0 = x0_ + x1_;  \
   xx1 = x0_ - x1_;  \
} while (0)


#define inv_butterfly_neg(xx0, xx1, w)  \
do  \
{  \
   cmplx_t x0_ = xx0;  \
   cmplx_t x1_ = xx1;  \
   cmplx_t t_ = x1_ * w;  \
   xx0 = x0_ - t_; /* NEG */   \
   xx1 = x0_ + t_; /* NEG */   \
} while (0)


// requires size divisible by 8
static void
new_ifft_layer(cmplx_t* xp, long blocks, long size, const cmplx_t* wtab)
{

  size /= 2;
  const cmplx_t* PGFFT_RESTRICT wtab1 = wtab + size;

  do
    {

      cmplx_t* PGFFT_RESTRICT xp0 = xp;
      cmplx_t* PGFFT_RESTRICT xp1 = xp + size;


      // first 4 butterflies
      inv_butterfly0(xp0[0], xp1[0]);
      inv_butterfly_neg(xp0[1], xp1[1], wtab1[-1]);
      inv_butterfly_neg(xp0[2], xp1[2], wtab1[-2]);
      inv_butterfly_neg(xp0[3], xp1[3], wtab1[-3]);

      // 4-way unroll
      for (long j = 4; j < size; j+= 4) {
         inv_butterfly_neg(xp0[j+0], xp1[j+0], wtab1[-(j+0)]);
         inv_butterfly_neg(xp0[j+1], xp1[j+1], wtab1[-(j+1)]);
         inv_butterfly_neg(xp0[j+2], xp1[j+2], wtab1[-(j+2)]);
         inv_butterfly_neg(xp0[j+3], xp1[j+3], wtab1[-(j+3)]);
      }

      xp += 2 * size;
    }
  while (--blocks != 0);
}

static void
new_ifft_first_two_layers(cmplx_t* xp, long blocks, const cmplx_t* wtab)
{
  // 4th root of unity
  cmplx_t w = wtab[1];

  do
    {
      cmplx_t u0 = xp[0];
      cmplx_t u1 = xp[1];
      cmplx_t u2 = xp[2];
      cmplx_t u3 = xp[3];

      cmplx_t v0 = u0 + u1;
      cmplx_t v1 = u0 - u1;
      cmplx_t v2 = u2 + u3;
      cmplx_t t  = u2 - u3;
      cmplx_t v3 = t * w;

      xp[0] = v0 + v2;
      xp[2] = v0 - v2;
      xp[1] = v1 - v3;  // NEG
      xp[3] = v1 + v3;  // NEG

      xp += 4;
    }
  while (--blocks != 0);
}


static void
new_ifft_base(cmplx_t* xp, long lgN, const vector<vector<cmplx_t>>& tab)
{
  if (lgN == 0) return;


  if (lgN == 1)
    {
      cmplx_t x0 = xp[0];
      cmplx_t x1 = xp[1];
      xp[0] = x0 + x1;
      xp[1] = x0 - x1;
      return;
    }


  long blocks = 1L << (lgN - 2);
  new_ifft_first_two_layers(xp, blocks, &tab[2][0]);
  blocks >>= 1;

  long size = 8;
  for (long j = 3; j <= lgN; j++, blocks >>= 1, size <<= 1)
    new_ifft_layer(xp, blocks, size, &tab[j][0]);
}

static
void new_ifft_short2(cmplx_t* yp, long yn, long lgN, const vector<vector<cmplx_t>>& tab);



static
void new_ifft_short1(cmplx_t* xp, long yn, long lgN, const vector<vector<cmplx_t>>& tab)

// Implements truncated inverse FFT interface, but with xn==yn.
// All computations are done in place.

{
  long N = 1L << lgN;

  if (yn == N && lgN <= PGFFT_NEW_FFT_THRESH)
    {
      // no truncation
      new_ifft_base(xp, lgN, tab);
      return;
    }

  // divide-and-conquer algorithm

  long half = N >> 1;

  if (yn <= half)
    {
      // X -> 2X
      for (long j = 0; j < yn; j++)
      	xp[j] = 2.0 * xp[j];

      new_ifft_short1(xp, yn, lgN - 1, tab);
    }
  else
    {
      cmplx_t* PGFFT_RESTRICT xp0 = xp;
      cmplx_t* PGFFT_RESTRICT xp1 = xp + half;
      const cmplx_t* PGFFT_RESTRICT wtab = &tab[lgN][0];

      new_ifft_short1(xp0, half, lgN - 1, tab);

      yn -= half;

      // X -> (2X, w*X)
      for (long j = yn; j < half; j++)
	{
	  cmplx_t x0 = xp0[j];
	  xp0[j] = 2.0 * x0;
	  xp1[j] = x0 * wtab[j];
	}

      new_ifft_short2(xp1, yn, lgN - 1, tab);

      // (X, Y) -> (X + Y/w, X - Y/w)
      {
	const cmplx_t* PGFFT_RESTRICT wtab1 = wtab + half;

	// DIRT: assumes yn is a multiple of 4
	inv_butterfly0(xp0[0], xp1[0]);
	inv_butterfly_neg(xp0[1], xp1[1], wtab1[-1]);
	inv_butterfly_neg(xp0[2], xp1[2], wtab1[-2]);
	inv_butterfly_neg(xp0[3], xp1[3], wtab1[-3]);
	for (long j = 4; j < yn; j+=4) {
	  inv_butterfly_neg(xp0[j+0], xp1[j+0], wtab1[-(j+0)]);
	  inv_butterfly_neg(xp0[j+1], xp1[j+1], wtab1[-(j+1)]);
	  inv_butterfly_neg(xp0[j+2], xp1[j+2], wtab1[-(j+2)]);
	  inv_butterfly_neg(xp0[j+3], xp1[j+3], wtab1[-(j+3)]);
	}
      }
    }
}



static
void new_ifft_short2(cmplx_t* xp, long yn, long lgN, const vector<vector<cmplx_t>>& tab)

// Implements truncated inverse FFT interface, but with xn==N.
// All computations are done in place.

{
  long N = 1L << lgN;

  if (yn == N && lgN <= PGFFT_NEW_FFT_THRESH)
    {
      // no truncation
      new_ifft_base(xp, lgN, tab);
      return;
    }

  // divide-and-conquer algorithm

  long half = N >> 1;

  if (yn <= half)
    {
      // X -> 2X
      for (long j = 0; j < yn; j++)
     	xp[j] = 2.0 * xp[j];
      // (X, Y) -> X + Y
      for (long j = yn; j < half; j++)
	xp[j] = xp[j] + xp[j + half];

      new_ifft_short2(xp, yn, lgN - 1, tab);

      // (X, Y) -> X - Y
      for (long j = 0; j < yn; j++)
	xp[j] = xp[j] - xp[j + half];
    }
  else
    {
      cmplx_t* PGFFT_RESTRICT xp0 = xp;
      cmplx_t* PGFFT_RESTRICT xp1 = xp + half;
      const cmplx_t* PGFFT_RESTRICT wtab = &tab[lgN][0];

      new_ifft_short1(xp0, half, lgN - 1, tab);

      yn -= half;


      // (X, Y) -> (2X - Y, w*(X - Y))
      for (long j = yn; j < half; j++)
	{
	  cmplx_t x0 = xp0[j];
	  cmplx_t x1 = xp1[j];
	  cmplx_t u = x0 - x1;
	  xp0[j] = x0 + u;
	  xp1[j] = u * wtab[j];
	}

      new_ifft_short2(xp1, yn, lgN - 1, tab);

      // (X, Y) -> (X + Y/w, X - Y/w)
      {
	const cmplx_t* PGFFT_RESTRICT wtab1 = wtab + half;

	// DIRT: assumes yn is a multiple of 4
	inv_butterfly0(xp0[0], xp1[0]);
	inv_butterfly_neg(xp0[1], xp1[1], wtab1[-1]);
	inv_butterfly_neg(xp0[2], xp1[2], wtab1[-2]);
	inv_butterfly_neg(xp0[3], xp1[3], wtab1[-3]);
	for (long j = 4; j < yn; j+=4) {
	  inv_butterfly_neg(xp0[j+0], xp1[j+0], wtab1[-(j+0)]);
	  inv_butterfly_neg(xp0[j+1], xp1[j+1], wtab1[-(j+1)]);
	  inv_butterfly_neg(xp0[j+2], xp1[j+2], wtab1[-(j+2)]);
	  inv_butterfly_neg(xp0[j+3], xp1[j+3], wtab1[-(j+3)]);
	}
      }
    }
}



static void 
new_ifft(cmplx_t* xp, long lgN, const vector<vector<cmplx_t>>& tab)
{
   long N = 1L << lgN;
   new_ifft_short1(xp, N, lgN, tab);
}


static void
compute_table(vector<vector<cmplx_t>>& tab, long k)
{
  if (k < 2) return;

  tab.resize(k+1);
  for (long s = 2; s <= k; s++) {
    long m = 1L << s;
    tab[s].resize(m/2);
    for (long j = 0; j < m/2; j++) {
      double angle = -((2 * M_PI) * (double(j)/double(m)));
      tab[s][j] = std::polar(1.0, angle);
    }
  }
}

static long 
RevInc(long a, long k)
{
   long j, m;

   j = k;
   m = 1L << (k-1);

   while (j && (m & a)) {
      a ^= m;
      m >>= 1;
      j--;
   }
   if (j) a ^= m;
   return a;
}

static void
BRC_init(long k, vector<long>& rev)
{
   long n = (1L << k);
   rev.resize(n);
   long i, j;
   for (i = 0, j = 0; i < n; i++, j = RevInc(j, k))
      rev[i] = j;
}



static
void BasicBitReverseCopy(cmplx_t *B, 
                         const cmplx_t *A, long k, const vector<long>& rev)
{
   long n = 1L << k;
   long i, j;

   for (i = 0; i < n; i++)
      B[rev[i]] = A[i];
}


static long
bluestein_precomp(long n, vector<cmplx_t>& powers, vector<cmplx_t>& Rb, 
                  vector<vector<cmplx_t>>& tab)
{
   // k = least k such that 2^k >= 2*n-1
   long k = 0;
   while ((1L << k) < 2*n-1) k++;

   compute_table(tab, k);

   powers.resize(n);
   powers[0] = 1;
   long i_sqr = 0;
   for (long i = 1; i < n; i++) {
      // i^2 = (i-1)^2 + 2*i-1
      i_sqr = (i_sqr + 2*i - 1) % (2*n);
      double angle = -((2 * M_PI) * (double(i_sqr)/double(2*n)));
      powers[i] = std::polar(1.0, angle);
   }

   long N = 1L << k;
   Rb.resize(N);
   for (long i = 0; i < N; i++) Rb[i] = 0;

   Rb[n-1] = 1;
   i_sqr = 0;
   for (long i = 1; i < n; i++) {
      // i^2 = (i-1)^2 + 2*i-1
      i_sqr = (i_sqr + 2*i - 1) % (2*n);
      double angle = (2 * M_PI) * (double(i_sqr)/double(2*n));
      Rb[n-1+i] = Rb[n-1-i] = std::polar(1.0, angle);
   }
  
   new_fft(&Rb[0], k, tab);

   return k;

}

static cmplx_t check_sum = 0;

static long
bluestein_comp(vector<cmplx_t>& a, 
                  long n, long k, const vector<cmplx_t>& powers, vector<cmplx_t>& Rb, 
                  const vector<vector<cmplx_t>>& tab)
{
   long N = 1L << k;

   vector<cmplx_t> x(N);

   for (long i = 0; i < n; i++)
      x[i] = a[i] * powers[i];

   for (long i = n; i < N; i++)
      x[i] = 0;

   new_fft(&x[0], k, tab);

   for (long i = 0; i < N; i++)
      x[i] *= Rb[i];

   new_ifft(&x[0], k, tab);

   double Ninv = 1/double(N);
   
   for (long i = 0; i < n; i++) 
      a[i] = x[n-1+i] * powers[i] * Ninv; 

   check_sum += a[0];
   
}


#define TIME_IT(t, action) \
do { \
   double _t0, _t1; \
   long _iter = 1; \
   long _cnt = 0; \
   do { \
      _t0 = NTL::GetTime(); \
      for (long _i = 0; _i < _iter; _i++) { action; _cnt++; } \
      _t1 = NTL::GetTime(); \
   } while ( _t1 - _t0 < 3 && (_iter *= 2)); \
   t = (_t1 - _t0)/_iter; \
} while(0)

int main()
{
#if 0
   long k = 6;
   long n = 64;

   vector<vector<cmplx_t>> tab;
   compute_table(tab, k);

   vector<cmplx_t> v(n);
   for (int i = 0; i < n; i++)
      v[i] = ((i+1)%6)*((i+1)%6); 


   //new_fft_base(&v[0], k, tab);
   new_fft(&v[0], k, tab);

   //new_ifft_base(&v[0], k, tab);
   new_ifft(&v[0], k, tab);

   for (long i = 0; i < n; i++) v[i] /= double(n);

#if 0
   vector<long> rev;
   BRC_init(k, rev);

   vector<cmplx_t> w(n);
   BasicBitReverseCopy(&w[0], &v[0], k, rev);
#endif

   for (int i = 0; i < n; i++)
      std::cout << v[i] << "\n";
#elif 0
   long n = 5;

   vector<cmplx_t> v(n);
   for (int i = 0; i < n; i++)
      v[i] = i*i+1;

   vector<cmplx_t> powers;
   vector<cmplx_t> Rb;
   vector<vector<cmplx_t>> tab;

   long k = bluestein_precomp(n, powers, Rb, tab);

   bluestein_comp(v, n, k, powers, Rb, tab);

   for (int i = 0; i < n; i++)
      std::cout << v[i] << "\n";
#else
   //long n = 28679;
   long n = 45551;

   vector<cmplx_t> v(n);
   for (long i = 0; i < n; i++)
      v[i] = NTL::RandomBnd(10)-5;

   vector<cmplx_t> w = v;

   vector<cmplx_t> powers;
   vector<cmplx_t> Rb;
   vector<vector<cmplx_t>> tab;

   long k = bluestein_precomp(n, powers, Rb, tab);

   double t;
   TIME_IT(t, (v=w, bluestein_comp(v, n, k, powers, Rb, tab))); 

   std::cout << t << "\n";
   std::cout << check_sum << "\n";


#endif
}
