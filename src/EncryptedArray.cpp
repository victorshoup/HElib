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
/* EncryptedArray.cpp - Data-movement operations on arrays of slots
 */
#include <algorithm>
#include "zzX.h"
#include "EncryptedArray.h"
#include "timing.h"
#include "clonedPtr.h"
#include "norms.h"

NTL_CLIENT

EncryptedArrayBase* buildEncryptedArray(const FHEcontext& context,
                                        const PAlgebraMod& alMod, const ZZX& G)
{
  if (alMod.getTag()==PA_cx_tag)
    return new EncryptedArrayCx(context, alMod.getCx());

  // By defualt use the 1st factor F0
  const ZZX& GG = NTL::IsZero(G)? alMod.getFactorsOverZZ()[0]: G;

  switch (alMod.getTag()) {
    case PA_GF2_tag: {
      return new EncryptedArrayDerived<PA_GF2>(context, conv<GF2X>(GG), alMod);
    }
    case PA_zz_p_tag: {
      zz_pBak bak; bak.save(); alMod.restoreContext();
      return new EncryptedArrayDerived<PA_zz_p>(context, conv<zz_pX>(GG), alMod);
    }
    default: return NULL;
  }
}


template<class type>
EncryptedArrayDerived<type>::EncryptedArrayDerived(
   const FHEcontext& _context, const RX& _G, const PAlgebraMod& alMod)
  : context(_context), tab(alMod.getDerived(type()))
{
  tab.mapToSlots(mappingData, _G); // Compute the base-G representation maps
}

// rotate ciphertext in dimension i by amt
template<class type>
void EncryptedArrayDerived<type>::rotate1D(Ctxt& ctxt, long i, long amt, bool dc) const
{
  FHE_TIMER_START;
  //OLD: assert(&context == &ctxt.getContext());
  helib::assertEq(&context, &ctxt.getContext(), "Context mismatch");
  //OLD: assert(i >= 0 && i < dimension());
  helib::assertInRange(i, 0l, dimension(), "i must be between 0 and dimension()");

  RBak bak; bak.save(); tab.restoreContext();

  const vector< vector< RX > >& maskTable = tab.getMaskTable();
  const PAlgebra& zMStar = getPAlgebra();
  long m = zMStar.getM();
  long ord = sizeOfDimension(i);
  amt %= ord;// DIRT: assumes division w/ remainder follows C++11 and C99 rules
  if (amt == 0) return;

  if (dc || nativeDimension(i)) { // native dimension or don't-care
    // For don't-care, we assume that any shifts "off the end" are zero
    ctxt.smartAutomorph(zMStar.genToPow(i, amt));
    return;
  }

  // more expensive "non-native" rotation

  if (amt < 0) amt += ord;  // Make sure amt is in the range [1,ord-1]
  //OLD: assert(maskTable[i].size() > 0);
  helib::assertTrue(maskTable[i].size() > 0, "Found non-positive sized mask table entry");

  ctxt.smartAutomorph(zMStar.genToPow(i, amt));
  // ctxt = \rho_i^{amt}(originalCtxt)

  Ctxt T(ctxt);
  T.smartAutomorph(zMStar.genToPow(i, -ord));
  // T = \rho_i^{amt-ord}(originalCtxt).
  // This strategy assumes is geared toward the
  // assumption that we have the key switch matrix 
  // for \rho_i^{-ord}

  const RX& mask = maskTable[i][amt];
  zzX mask_poly = balanced_zzX(mask);
  double sz = embeddingLargestCoeff(mask_poly, zMStar);
  DoubleCRT m1(mask_poly, context, 
               ctxt.getPrimeSet() | T.getPrimeSet());
  // m1 will be used to multiply both ctxt and T
  
  // Compute ctxt = ctxt*m1 + T - T*m1
  ctxt.multByConstant(m1, sz);
  ctxt += T;
  T.multByConstant(m1, sz);
  ctxt -= T;
}

// Shift k positions along the i'th dimension with zero fill.
// Negative shift amount denotes shift in the opposite direction.
template<class type>
void EncryptedArrayDerived<type>::shift1D(Ctxt& ctxt, long i, long k) const
{
  FHE_TIMER_START;
  const PAlgebra& al = getPAlgebra();

  const vector< vector< RX > >& maskTable = tab.getMaskTable();

  RBak bak; bak.save(); tab.restoreContext();

  //OLD: assert(&context == &ctxt.getContext());
  helib::assertEq(&context, &ctxt.getContext(), "Context mismatch");
  //OLD: assert(i >= 0 && i < long(al.numOfGens()));
  helib::assertInRange(i, 0l, (long)(al.numOfGens()), "i must be non-negative and less than the PAlgebra's generator count");

  long ord = al.OrderOf(i);

  if (k <= -ord || k >= ord) {
    ctxt.multByConstant(to_ZZ(0));
    return;
  }

  // Make sure amt is in the range [1,ord-1]
  long amt = k % ord;
  if (amt == 0) return;
  if (amt < 0) amt += ord;

  RX mask = maskTable[i][ord-amt];

  long val;
  if (k < 0)
    val = al.genToPow(i, amt-ord);
  else {
    mask = 1 - mask;
    val = al.genToPow(i, amt);
  }
  ctxt.multByConstant(balanced_zzX(mask));   // zero out slots where mask=0
  ctxt.smartAutomorph(val);  // shift left by val
  FHE_TIMER_STOP;
}


// NOTE: masking depth: if there are N dimensions, and if for i = 1..N
// we define c_i = 1 if dimension i is bad and 0 o/w, then the masking
// depth is N - 1 + \sum_{i=1} c_i.  

template<class type>
void EncryptedArrayDerived<type>::rotate(Ctxt& ctxt, long amt) const
{
  FHE_TIMER_START;

  const PAlgebra& al = getPAlgebra();

  const vector< vector< RX > >& maskTable = tab.getMaskTable();

  RBak bak; bak.save(); tab.restoreContext();

  //OLD: assert(&context == &ctxt.getContext());
  helib::assertEq(&context, &ctxt.getContext(), "Context mismatch");

  // Simple case: just one generator
  if (al.numOfGens()==1) { // VJS: bug fix: <= must be ==
    rotate1D(ctxt, 0, amt);
    return;
  }

  // Make sure that amt is in [1,nslots-1]
  amt %= (long) al.getNSlots();
  if (amt == 0) { return; }
  if (amt < 0) amt += al.getNSlots();

  // rotate the ciphertext, one dimension at a time
  long i = al.numOfGens()-1;
  long v = al.coordinate(i, amt);
  RX mask = maskTable[i][v];
  Ctxt tmp(ctxt.getPubKey());
  const RXModulus& PhimXmod = tab.getPhimXMod();

  // optimize for the common case where the last generator has order in
  // Zm*/(p) different than its order in Zm*. In this case we can combine
  // the rotate1D relative to this generator with the masking after the
  // rotation. This saves one mult-by-constant, since we use the same mask
  // inside rotate1D as in the loop below.

  if (al.SameOrd(i) || v==0) rotate1D(ctxt, i, v); // no need to optimize
  else {


    long ord = al.OrderOf(i);

    ctxt.smartAutomorph(al.genToPow(i, v));
    // ctxt = \rho_i^{v}(originalCtxt)

    tmp = ctxt;
    tmp.smartAutomorph(al.genToPow(i, -ord));
    // tmp = \rho_i^{v-ord}(originalCtxt).
    // This strategy assumes is geared toward the
    // assumption that we have the key switch matrix 
    // for \rho_i^{-ord}

    zzX mask_poly = balanced_zzX(mask);
    double sz = embeddingLargestCoeff(mask_poly, al);

    DoubleCRT m1(mask_poly, context, 
		 ctxt.getPrimeSet() | tmp.getPrimeSet());
    // m1 will be used to multiply both ctxt and tmp
    
    // Compute ctxt = ctxt*m1, tmp = tmp*(1-m1)
    ctxt.multByConstant(m1, sz);

    Ctxt tmp1(tmp);
    tmp1.multByConstant(m1, sz);
    tmp -= tmp1;


    // apply rotation relative to next generator before combining the parts
    --i;
    v = al.coordinate(i, amt);
    rotate1D(ctxt, i, v); 
    rotate1D(tmp, i, v+1);
    ctxt += tmp;         // combine the two parts

    if (i <= 0) { return; }  // no more generators

    mask = ((mask * (maskTable[i][v] - maskTable[i][v+1])) % PhimXmod)
             + maskTable[i][v+1];  // update the mask for next iteration
  }

  // Handle rotation relative to all the other generators (if any)
  for (i--; i >= 0; i--) {
    v = al.coordinate(i, amt);

    zzX mask_poly = balanced_zzX(mask);

    tmp = ctxt;
    tmp.multByConstant(mask_poly); // only the slots in which mask=1
    ctxt -= tmp;                   // only the slots in which mask=0

    rotate1D(tmp, i, v); 
    rotate1D(ctxt, i, v+1);
    ctxt += tmp;
    if (i>0) {
      mask = ((mask * (maskTable[i][v] - maskTable[i][v+1])) % PhimXmod)
             + maskTable[i][v+1];  // update the mask for next iteration
    }
  }
  FHE_TIMER_STOP;
}

template<class type>
void EncryptedArrayDerived<type>::shift(Ctxt& ctxt, long k) const
{
  FHE_TIMER_START;


  const PAlgebra& al = getPAlgebra();

  const vector< vector< RX > >& maskTable = tab.getMaskTable();

  RBak bak; bak.save(); tab.restoreContext();

  //OLD: assert(&context == &ctxt.getContext());
  helib::assertEq(&context, &ctxt.getContext(), "Context mismatch");

  // Simple case: just one generator
  if (al.numOfGens()==1) {
    shift1D(ctxt, 0, k);
    return;
  }

  long nSlots = al.getNSlots();

  // Shifting by more than the number of slots gives an all-zero cipehrtext
  if (k <= -nSlots || k >= nSlots) {
    ctxt.multByConstant(to_ZZ(0));
    return;
  }

  // Make sure that amt is in [1,nslots-1]
  long amt = k % nSlots;
  if (amt == 0) return;
  if (amt < 0) amt += nSlots;

  // rotate the ciphertext, one dimension at a time
  long i = al.numOfGens()-1;
  long v = al.coordinate(i, amt);
  RX mask = maskTable[i][v];
  Ctxt tmp(ctxt.getPubKey());
  const RXModulus& PhimXmod = tab.getPhimXMod();

  rotate1D(ctxt, i, v);
  for (i--; i >= 0; i--) {
    v = al.coordinate(i, amt);

    zzX mask_poly = balanced_zzX(mask);

    tmp = ctxt;
    tmp.multByConstant(mask_poly); // only the slots in which mask=1
    ctxt -= tmp;                   // only the slots in which mask=0
    if (i>0) {
      rotate1D(ctxt, i, v+1);
      rotate1D(tmp, i, v); 
      ctxt += tmp;                    // combine the two parts

      mask = ((mask * (maskTable[i][v] - maskTable[i][v+1])) % PhimXmod)
             + maskTable[i][v+1];  // update the mask before next iteration
    }
    else { // i == 0
      if (k < 0) v -= al.OrderOf(0);
      shift1D(tmp, 0, v);
      shift1D(ctxt, 0, v+1);
      ctxt += tmp;
    } 
  }
  FHE_TIMER_STOP;
}

template<class type>
void EncryptedArrayDerived<type>::encode(ZZX& ptxt, const vector< RX >& array) const
{
  RX pp;
  tab.embedInSlots(pp, array, mappingData); 

  // NOTE: previous version was
  //   ptxt = conv<ZZX>(pp);
  // which did not do balanced remainders at all
  zzX pp1 = balanced_zzX(pp);
  convert(ptxt, pp1);
}

template<class type>
void EncryptedArrayDerived<type>::decode(vector< RX >& array, const ZZX& ptxt) const
{
  FHE_TIMER_START;
  RX pp;
  conv(pp, ptxt);
  tab.decodePlaintext(array, pp, mappingData); 
  FHE_TIMER_STOP;
}

template<class type>
void EncryptedArrayDerived<type>::encode(RX& ptxt, const vector< RX >& array) const
{
  tab.embedInSlots(ptxt, array, mappingData); 
}

template<class type>
void EncryptedArrayDerived<type>::decode(vector< RX >& array, const RX& ptxt) const
{
  tab.decodePlaintext(array, ptxt, mappingData); 
}

template<class type>
void EncryptedArrayDerived<type>::encode(ZZX& ptxt, const PlaintextArray& array) const
{
  RBak bak; bak.save(); tab.restoreContext();
  encode(ptxt, array.getData<type>());
}

template<class type>
void EncryptedArrayDerived<type>::decode(PlaintextArray& array, const ZZX& ptxt) const
{
  RBak bak; bak.save(); tab.restoreContext();
  decode(array.getData<type>(), ptxt);
}


template<class type>
void EncryptedArrayDerived<type>::encodeUnitSelector(zzX& ptxt, long i) const
{
  //OLD: assert(i >= 0 && i < (long)getPAlgebra().getNSlots());
  helib::assertInRange(i, 0l, (long)getPAlgebra().getNSlots(), "i must be non-negative and less than the PAlgebra's slot count");
  RBak bak; bak.save(); tab.restoreContext();
  RX res;
  div(res, tab.getPhimXMod(), tab.getFactors()[i]); 
  mul(res, res, tab.getCrtCoeffs()[i]);
  convert(ptxt, res);
}

template<class type>
void EncryptedArrayDerived<type>::encode(zzX& ptxt, const vector< RX >& array) const
{
  RX pp;
  tab.embedInSlots(pp, array, mappingData); 

  // NOTE: previous version was
  //   convert(ptxt, pp);
  // which did not do properly balanced remainders in some cases
  ptxt = balanced_zzX(pp);
}

template<class type>
void EncryptedArrayDerived<type>::encode(zzX& ptxt, const PlaintextArray& array) const
{
  RBak bak; bak.save(); tab.restoreContext();
  encode(ptxt, array.getData<type>());
}

template<class type>
void EncryptedArrayDerived<type>::decode(vector< RX >& array, const NTL::Vec<long>& ptxt) const
{
  FHE_TIMER_START;
  RX pp;
  convert(pp, ptxt);
  tab.decodePlaintext(array, pp, mappingData); 
  FHE_TIMER_STOP;
}

template<class type>
void EncryptedArrayDerived<type>::decode(PlaintextArray& array, const NTL::Vec<long>& ptxt) const
{
  RBak bak; bak.save(); tab.restoreContext();
  decode(array.getData<type>(), ptxt);
}


// this routine generates a "random" normal element and initializes a
// matrix mapping from polynomial to normal basis and its inverse. It
// uses a randomized algorithm to find the normal basis, but we init
// the PRG seed deterministically to ensure that we always get the
// same one (for a given set of parameters)

template<class type>
void EncryptedArrayDerived<type>::initNormalBasisMatrix() const
{
  RandomState state;
  SetSeed(to_ZZ(1));
  do {
    typename Lazy< Pair< Mat<R>, Mat<R> > >::Builder 
      builder(normalBasisMatrices); 

    if (!builder()) break;

    RBak bak; bak.save(); restoreContext();
    REBak ebak; ebak.save(); restoreContextForG();

    long d = RE::degree();
    long p = getPAlgebra().getP();
    long r = tab.getR();

    // compute change of basis matrix CB
    mat_R CB;
    CB.SetDims(d, d);
    RE normal_element;
    RE H;
    bool got_it = false;

    H = power(conv<RE>(RX(1, 1)), p);
    

    do {
      NTL::random(normal_element);
   
      RE pow;
      pow = normal_element; 
      VectorCopy(CB[0], rep(pow), d);
      for (long i = 1; i < d; i++) {
        pow = eval(rep(pow), H);
        VectorCopy(CB[i], rep(pow), d);
      }

      Mat<ZZ> CB1;
      conv(CB1, CB);

      {
         zz_pBak bak1; bak1.save(); zz_p::init(p);
         Mat<zz_p> CB2;
         conv(CB2, CB1);
         got_it = determinant(CB2) != 0;
      }
    } while (!got_it);

    Mat<R> CBi;
    ppInvert(CBi, CB, p, r);

    UniquePtr< Pair< Mat<R>, Mat<R> > > ptr;
    ptr.make(CB, CBi);
    builder.move(ptr);
  } while(0);
}



// Other functions...

void runningSums(const EncryptedArray& ea, Ctxt& ctxt)
{
  long n = ea.size();

  long shamt = 1;
  while (shamt < n) {
    Ctxt tmp = ctxt;
    ea.shift(tmp, shamt);
    ctxt += tmp; // ctxt = ctxt + (ctxt >> shamt)
    shamt = 2*shamt;
  }
}

void totalSums(const EncryptedArray& ea, Ctxt& ctxt)
{
  long n = ea.size();

  if (n == 1) return;

  Ctxt orig = ctxt;

  long k = NumBits(n);
  long e = 1;

  for (long i = k-2; i >= 0; i--) {
    Ctxt tmp1 = ctxt;
    ea.rotate(tmp1, e);
    ctxt += tmp1; // ctxt = ctxt + (ctxt >>> e)
    e = 2*e;

    if (bit(n, i)) {
      Ctxt tmp2 = orig;
      ea.rotate(tmp2, e);
      ctxt += tmp2; // ctxt = ctxt + (orig >>> e)
                    // NOTE: we could have also computed
                    // ctxt =  (ctxt >>> e) + orig, however,
                    // this would give us greater depth/noise
      e += 1;
    }
  }
}




// Linearized polynomials.
// L describes a linear map M by describing its action on the standard
// power basis: M(x^j mod G) = (L[j] mod G), for j = 0..d-1.  
// The result is a coefficient vector C for the linearized polynomial
// representing M: a polynoamial h in Z/(p^r)[X] of degree < d is sent to
//
//    M(h(X) \bmod G)= \sum_{i=0}^{d-1}(C[j] \cdot h(X^{p^j}))\bmod G).
template<class type> void
EncryptedArrayDerived<type>::buildLinPolyCoeffs(vector<ZZX>& C, 
						const vector<ZZX>& L) const
{
  RBak bak; bak.save(); restoreContext();
  vector<RX> CC, LL;
  convert(LL, L);
  buildLinPolyCoeffs(CC, LL);
  convert(C, CC);
}

template<class type> void
EncryptedArrayDerived<type>::buildLinPolyCoeffs(vector<RX>& C, 
						const vector<RX>& L) const
{
  FHE_TIMER_START;

  RBak bak; bak.save(); restoreContext();  // the NTL context for mod p^r
  REBak ebak; ebak.save(); restoreContextForG(); // The NTL context for mod G

  do {
    typename Lazy< Mat<RE> >::Builder builder(linPolyMatrix);
    if (!builder()) break;

    FHE_NTIMER_START(buildLinPolyCoeffs_invert);

   
    long p = getPAlgebra().getP();
    long r = tab.getR();

    Mat<RE> M1;
    // build d x d matrix, d is taken from the current NTL context for G
    buildLinPolyMatrix(M1, p);
    Mat<RE> M2;
    ppInvert(M2, M1, p, r); // invert modulo prime-power p^r

    UniquePtr< Mat<RE> > ptr;
    ptr.make(M2);
    builder.move(ptr);
  } while (0);

  Vec<RE> CC, LL;
  convert(LL, L);
  mul(CC, LL, *linPolyMatrix);
  convert(C, CC);
}


// Apply the same linear transformation to all the slots.
// C[0...d-1] is the output of ea.buildLinPolyCoeffs
void applyLinPoly1(const EncryptedArray& ea, Ctxt& ctxt, const vector<ZZX>& C)
{
  //OLD: assert(&ea.getContext() == &ctxt.getContext());
  helib::assertEq(&ea.getContext(), &ctxt.getContext(), "Context mismatch");
  long d = ea.getDegree();
  //OLD: assert(d == lsize(C));
  helib::assertEq(d, lsize(C), "ea's degree does not match the size of C");

  long nslots = ea.size();

  vector<ZZX> encodedC(d);
  for (long j = 0; j < d; j++) {
    vector<ZZX> v(nslots); // all the slots of v equal C[j]
    for (long i = 0; i < nslots; i++) v[i] = C[j];
    ea.encode(encodedC[j], v);
  }

  applyLinPolyLL(ctxt, encodedC, ea.getDegree());
}


// Apply different transformations to different slots. Each row in
// the matrix Cvec[0...nslots-1][0...d-1] is a length-d vector which
// is the output of ea.buildLinPolyCoeffs
void applyLinPolyMany(const EncryptedArray& ea, Ctxt& ctxt, 
                      const vector< vector<ZZX> >& Cvec)
{
  //OLD: assert(&ea.getContext() == &ctxt.getContext());
  helib::assertEq(&ea.getContext(), &ctxt.getContext(), "Context mismatch");
  long d = ea.getDegree();
  long nslots = ea.size();

  //OLD: assert(nslots == lsize(Cvec));
  helib::assertEq(nslots, lsize(Cvec), "Number of slots does not match size of Cvec");
  for (long i = 0; i < nslots; i++) {
    //OLD: assert(d == lsize(Cvec[i]));
    helib::assertEq(d, lsize(Cvec[i]), "Found entry of Cvec with size unequal to degree of ea");
  }

  vector<ZZX> encodedC(d);
  for (long j = 0; j < d; j++) { // encodedC[j] encodes j'th column in Cvec
    vector<ZZX> v(nslots);       // copy j'th column to v
    for (long i = 0; i < nslots; i++) v[i] = Cvec[i][j];
    ea.encode(encodedC[j], v);   // then encode it
  }

  applyLinPolyLL(ctxt, encodedC, ea.getDegree());
}

// A low-level variant: encodedCoeffs has all the linPoly coeffs encoded
// in slots; different transformations can be encoded in different slots
template<class P>
void applyLinPolyLL(Ctxt& ctxt, const vector<P>& encodedC, long d)
{
  //OLD: assert(d == lsize(encodedC));
  helib::assertEq(d, lsize(encodedC), "d does not match size of encodedC");

  ctxt.cleanUp();  // not sure, but this may be a good idea

  Ctxt tmp(ctxt);

  ctxt.multByConstant(encodedC[0]);
  for (long j = 1; j < d; j++) {
    Ctxt tmp1(tmp);
    tmp1.frobeniusAutomorph(j);
    tmp1.multByConstant(encodedC[j]);
    ctxt += tmp1;
  }
}
template void applyLinPolyLL(Ctxt& ctxt, const vector<zzX>& encodedC, long d);
template void applyLinPolyLL(Ctxt& ctxt, const vector<ZZX>& encodedC, long d);
template void applyLinPolyLL(Ctxt& ctxt, const vector<DoubleCRT>& encodedC, long d);

/****************** End linear transformation code ******************/
/********************************************************************/


// PlaintextArray


template<class type>
class rotate_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa, long k)
  {
    PA_BOILER

    vector<RX> tmp(n); 

    for (long i = 0; i < n; i++)
      tmp[((i+k)%n + n)%n] = data[i];

    data = tmp;
  }
};

void rotate(const EncryptedArray& ea, PlaintextArray& pa, long k)
{
  ea.dispatch<rotate_pa_impl>(pa, k); 
}

//=============================================================================

template<class type>
class shift_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa, long k)
  {
    PA_BOILER

    for (long i = 0; i < n; i++)
      if (i + k >= n || i + k < 0)
        clear(data[i]);

    rotate_pa_impl<type>::apply(ea, pa, k); 
  }
};

void shift(const EncryptedArray& ea, PlaintextArray& pa, long k)
{
  ea.dispatch<shift_pa_impl>(pa, k); 
}

//=============================================================================

template<class type>
class encode_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea,
                    PlaintextArray& pa, const vector<long>& array)
  {
    PA_BOILER

    //OLD: assert(lsize(array) == n);
    helib::assertEq(lsize(array), n, "Size of array does not match n");
    convert(data, array);
  }

  static void apply(const EncryptedArrayDerived<type>& ea,
                    PlaintextArray& pa, const vector<ZZX>& array)
  {
    PA_BOILER

    //OLD: assert(lsize(array) == n);
    helib::assertEq(lsize(array), n, "Size of array does not match n");
    convert(data, array);
    for (long i = 0; i < n; i++) {
      helib::assertTrue(deg(data[i]) < d, "Found data entry with too-large degree");
    }
  }

};



void encode(const EncryptedArray& ea, PlaintextArray& pa, const vector<long>& array)
{
  ea.dispatch<encode_pa_impl>(pa, array); 
}

void encode(const EncryptedArray& ea, PlaintextArray& pa, const vector<ZZX>& array)
{
  ea.dispatch<encode_pa_impl>(pa, array); 
}

void encode(const EncryptedArray& ea, PlaintextArray& pa, long val)
{
   long n = ea.size();
   vector<long> array;
   array.resize(n);
   for (long i = 0; i < n; i++) array[i] = val;
   encode(ea, pa, array);
}

void encode(const EncryptedArray& ea, PlaintextArray& pa, const ZZX& val)
{
   long n = ea.size();
   vector<ZZX> array;
   array.resize(n);
   for (long i = 0; i < n; i++) array[i] = val;
   encode(ea, pa, array);
}

//=============================================================================

template<class type>
class random_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa)
  {
    PA_BOILER

    for (long i = 0; i < n; i++)
      random(data[i], d);
  }
}; 


void random(const EncryptedArray& ea, PlaintextArray& pa)
{
  ea.dispatch<random_pa_impl>(pa); 
}

//=============================================================================

template<class type>
class decode_pa_impl {
public:
  PA_INJECT(type)

  template<class T>
  static void apply(const EncryptedArrayDerived<type>& ea, 
    vector<T>& array, const PlaintextArray& pa)
  {
    CPA_BOILER

    convert(array, data);
  }

}; 


void decode(const EncryptedArray& ea, vector<long>& array, const PlaintextArray& pa)
{
  ea.dispatch<decode_pa_impl>(array, pa); 
}


void decode(const EncryptedArray& ea, vector<ZZX>& array, const PlaintextArray& pa)
{
  ea.dispatch<decode_pa_impl>(array, pa); 
}

//=============================================================================

template<class type>
class equals_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, bool& res, 
    const PlaintextArray& pa, const  PlaintextArray& other)
  {
    CPA_BOILER

    const vector<RX>& odata = other.getData<type>(); 
    res = (data == odata);
  }


  static void apply(const EncryptedArrayDerived<type>& ea, bool& res, 
    const PlaintextArray& pa, const vector<long>& other)
  {
    CPA_BOILER

    vector<RX> odata;
    convert(odata, other);
    res = (data == odata);
  }


  static void apply(const EncryptedArrayDerived<type>& ea, bool& res,
    const PlaintextArray& pa, const vector<ZZX>& other)
  {
    CPA_BOILER

    vector<RX> odata;
    convert(odata, other);
    res = (data == odata);
  }

};



bool equals(const EncryptedArray& ea, const PlaintextArray& pa, const PlaintextArray& other)
{
  bool res;
  ea.dispatch<equals_pa_impl>(res, pa, other); 
  return res;
}

bool equals(const EncryptedArray& ea, const PlaintextArray& pa, const vector<long>& other)
{
  bool res;
  ea.dispatch<equals_pa_impl>(res, pa, other); 
  return res;
}


bool equals(const EncryptedArray& ea, const PlaintextArray& pa, const vector<ZZX>& other)
{
  bool res;
  ea.dispatch<equals_pa_impl>(res, pa, other); 
  return res;
}

//=============================================================================

template<class type>
class add_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa, 
    const PlaintextArray& other)
  {
    PA_BOILER

    const vector<RX>& odata = other.getData<type>(); 

    for (long i = 0; i < n; i++)
      data[i] += odata[i];
  }
}; 


void add(const EncryptedArray& ea, PlaintextArray& pa, const PlaintextArray& other)
{
  ea.dispatch<add_pa_impl>(pa, other); 
}

//=============================================================================

template<class type>
class sub_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa, 
    const PlaintextArray& other)
  {
    PA_BOILER

    const vector<RX>& odata = other.getData<type>(); 

    for (long i = 0; i < n; i++)
      data[i] -= odata[i];
  }
}; 


void sub(const EncryptedArray& ea, PlaintextArray& pa, const PlaintextArray& other)
{
  ea.dispatch<sub_pa_impl>(pa, other); 
}

//=============================================================================

template<class type>
class mul_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa, 
    const PlaintextArray& other)
  {
    PA_BOILER

    const vector<RX>& odata = other.getData<type>(); 

    for (long i = 0; i < n; i++)
      data[i] = (data[i] * odata[i]) % G;
  }
}; 


void mul(const EncryptedArray& ea, PlaintextArray& pa, const PlaintextArray& other)
{
  ea.dispatch<mul_pa_impl>(pa, other); 
}

//=============================================================================

template<class type>
class negate_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa) 
  {
    PA_BOILER

    for (long i = 0; i < n; i++)
      NTL::negate(data[i], data[i]);
  }
}; 


void negate(const EncryptedArray& ea, PlaintextArray& pa)
{
  ea.dispatch<negate_pa_impl>(pa); 
}

//=============================================================================

template<class type>
class frobeniusAutomorph_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa, long j)
  {
    PA_BOILER

    long p = ea.getPAlgebra().getP();

    j = mcMod(j, d);
    RX H = PowerMod(RX(1, 1), power_ZZ(p, j), G);

    for (long i = 0; i < n; i++)
      data[i] = CompMod(data[i], H, G);
  }

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa,
    const Vec<long>& vec) 
  {
    PA_BOILER

    //OLD: assert(vec.length() == n);
    helib::assertEq(vec.length(), n, "vec has incorrect length");

    long p = ea.getPAlgebra().getP();

    for (long i = 0; i < n; i++) {
      long j = mcMod(vec[i], d);
      RX H = PowerMod(RX(1, 1), power_ZZ(p, j), G);
      data[i] = CompMod(data[i], H, G);
    }
  }
};




void frobeniusAutomorph(const EncryptedArray& ea, PlaintextArray& pa, long j)
{
  ea.dispatch<frobeniusAutomorph_pa_impl>(pa, j); 
}


void frobeniusAutomorph(const EncryptedArray& ea, PlaintextArray& pa, const Vec<long>& vec)
{
  ea.dispatch<frobeniusAutomorph_pa_impl>(pa, vec); 
}

void power(const EncryptedArray& ea, PlaintextArray& pa, long e)
{
  if (e<=1) return;

  PlaintextArray pwr = pa; // holds x^{2^i} in i+1'st iteration
  encode(ea, pa, 1L); // set pa =1 in every slot
  while(e > 0) {
    if (e & 1)
      mul(ea, pa, pwr); // multiply if needed
    mul(ea, pwr, pwr);  // squre
    e >>= 1;
  }
}

//=============================================================================

template<class type>
class applyPerm_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, PlaintextArray& pa,
    const Vec<long>& pi) 
  {
    PA_BOILER

    //OLD: assert(pi.length() == n);
    helib::assertEq(pi.length(), n, "pi has incorrect length");

    vector<RX> tmp;
    tmp.resize(n);
    for (long i = 0; i < n; i++)
      tmp[i] = data[pi[i]];

    data = tmp;
  }
};




void applyPerm(const EncryptedArray& ea, PlaintextArray& pa, const Vec<long>& pi)
{
  ea.dispatch<applyPerm_pa_impl>(pa, pi); 
}

//=============================================================================

template<class type>
class print_pa_impl {
public:
  PA_INJECT(type)

  static void apply(const EncryptedArrayDerived<type>& ea, 
    ostream& s, const PlaintextArray& pa)
  {
    CPA_BOILER


    if (n == 0) 
      s << "[]";
    else {
      if (IsZero(data[0])) s << "[[0]";
      else                 s << "[" << data[0];
      for (long i = 1; i < lsize(data); i++)
        if (IsZero(data[i])) s << " [0]";
	else                 s << " " << data[i];
      s << "]";
    }
  }

}; 


void print(const EncryptedArray& ea, ostream& s, const PlaintextArray& pa)
{
  ea.dispatch<print_pa_impl>(s, pa); 
}


// Explicit instantiation

template class EncryptedArrayDerived<PA_GF2>;
template class EncryptedArrayDerived<PA_zz_p>;


template class PlaintextArrayDerived<PA_GF2>;
template class PlaintextArrayDerived<PA_zz_p>;

