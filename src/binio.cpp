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
#include "binio.h"
#include "assertions.h"
#include <cstring>
#include <sys/types.h>
// NOTE: including sys/types.h for the purpose of bringing in 
// the byte order macros in a platform-independent way.

NTL_CLIENT
/* Some utility functions for binary IO */

int readEyeCatcher(istream& str, const char * expect)
{
  char eye[BINIO_EYE_SIZE];
  str.read(eye, BINIO_EYE_SIZE); 
  return memcmp(eye, expect, BINIO_EYE_SIZE);
}

void writeEyeCatcher(ostream& str, const char* eyeStr)
{
  char eye[BINIO_EYE_SIZE];
  memcpy(eye, eyeStr, BINIO_EYE_SIZE);
  str.write(eye, BINIO_EYE_SIZE);
}

// compile only 64-bit (-m64) therefore long must be at least 64-bit
long read_raw_int(istream& str)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  long result = 0;
  str.read((char*)&result, BINIO_64BIT);
  return result;
#else
  long result = 0;
  char byte;

  for(long i=0; i<BINIO_64BIT; i++){
    str.read(&byte, 1); // read a byte
    result |= (static_cast<long>(byte)&0xff) << i*8; // must be in little endian
  }

  return result;
#endif
}

int read_raw_int32(istream& str)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  int result = 0;
  str.read((char*)&result, BINIO_32BIT);
  return result;
#else
  int result = 0;
  char byte;

  for(long i=0; i<BINIO_32BIT; i++){
    str.read(&byte, 1); // read a byte
    result |= (static_cast<long>(byte)&0xff) << i*8; // must be in little endian
  }

  return result;
#endif
}

// compile only 64-bit (-m64) therefore long must be at least 64-bit
void write_raw_int(ostream& str, long num)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  str.write((const char*)&num, BINIO_64BIT);
#else
  char byte;

  for(long i=0; i<BINIO_64BIT; i++){
    byte = num >> 8*i; // serializing in little endian
    str.write(&byte, 1);  // write byte out
  }
#endif
}

void write_raw_int32(ostream& str, int num)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  str.write((const char*)&num, BINIO_32BIT);
#else
  char byte;

  for(long i=0; i<BINIO_32BIT; i++){
    byte = num >> 8*i; // serializing in little endian
    str.write(&byte, 1);  // write byte out
  }
#endif
}

void write_ntl_vec_long(ostream& str, const vec_long& vl, long intSize)
{
  //OLD: assert(intSize == BINIO_64BIT || intSize == BINIO_32BIT);
  helib::assertTrue<helib::InvalidArgument>(intSize == BINIO_64BIT || intSize == BINIO_32BIT, "intSize must be 32 or 64 bit for binary IO");
  write_raw_int32(str, vl.length());
  write_raw_int32(str, intSize);

  if (intSize == BINIO_64BIT) {
    for(long i=0; i<vl.length(); i++){
      write_raw_int(str, vl[i]);
    }
  } else {
    for(long i=0; i<vl.length(); i++){
      write_raw_int32(str, vl[i]);
    }
  }
}

void read_ntl_vec_long(istream& str, vec_long& vl)
{
  int sizeOfVL = read_raw_int32(str);
  int intSize  = read_raw_int32(str);
  //OLD: assert(intSize == BINIO_64BIT || intSize == BINIO_32BIT);
  helib::assertTrue<helib::InvalidArgument>(intSize == BINIO_64BIT || intSize == BINIO_32BIT, "intSize must be 32 or 64 bit for binary IO");

  // Remember to check and increase Vec before trying to fill it.
  if(vl.length() < sizeOfVL){
    vl.SetLength(sizeOfVL);
  }

  if (intSize == BINIO_64BIT) {
    for(long i=0; i<sizeOfVL; i++){
      vl[i] = read_raw_int(str);
    }
  } else {
    for(long i=0; i<sizeOfVL; i++){
      vl[i] = read_raw_int32(str);
    }
  }
}

void write_raw_double(ostream& str, const double d)
{
  // FIXME: this is not portable: 
  //  * we might have sizeof(long) < sizeof(double)
  //  * we also don't know if the bit layout is really compatible
  const long *pd = reinterpret_cast<const long*>(&d);
  write_raw_int(str, *pd);
}

double read_raw_double(istream& str)
{
  // FIXME: see FIXME for write_raw_double
  long d = read_raw_int(str);
  double* pd = reinterpret_cast<double*>(&d);
  return *pd;
}

void write_raw_xdouble(ostream& str, const xdouble xd)
{
  double m = xd.mantissa();
  long e = xd.exponent();
  write_raw_double(str, m);
  write_raw_int(str, e);
}

xdouble read_raw_xdouble(istream& str)
{
  double m = read_raw_double(str);
  long e = read_raw_int(str);
  return xdouble(m,e);
}

void write_raw_ZZ(ostream& str, const ZZ& zz)
{
  long noBytes = NumBytes(zz);
  //OLD: assert(noBytes > 0);
  helib::assertTrue<helib::InvalidArgument>(noBytes > 0, "Number of bytes to write must be non-negative");
  unsigned char zzBytes[noBytes];
  BytesFromZZ(zzBytes, zz, noBytes); // From ZZ.h
  write_raw_int(str, noBytes); 
  // TODO - ZZ appears to be endian agnostic
  str.write(reinterpret_cast<char*>(zzBytes), noBytes); 
}

void read_raw_ZZ(istream& str, ZZ& zz)
{
  long noBytes = read_raw_int(str);
  //OLD: assert(noBytes > 0);
  helib::assertTrue<helib::InvalidArgument>(noBytes > 0, "Number of bytes to write must be non-negative");
  unsigned char zzBytes[noBytes];
  // TODO - ZZ appears to be endian agnostic
  str.read(reinterpret_cast<char*>(zzBytes), noBytes); 
  zz = ZZFromBytes(zzBytes, noBytes);
}


// FIXME: there is some repetitive code here.
// We should think about a better overloading strategy for
// read/write_raw_vector to avoid this.

template<> void read_raw_vector<long>(istream& str, vector<long>& v)
{

  long sz = read_raw_int(str);
  v.resize(sz); // Make space in vector

  for(long i=0; i<sz; i++)
    v[i] = read_raw_int(str); 
    
};

template<> void write_raw_vector<long>(ostream& str, const vector<long>& v)
{
  long sz = v.size();  
  write_raw_int(str, sz); 

  for(long n: v)
    write_raw_int(str, n); 
};


template<> void read_raw_vector<double>(istream& str, vector<double>& v)
{

  long sz = read_raw_int(str);
  v.resize(sz); // Make space in vector

  for(long i=0; i<sz; i++)
    v[i] = read_raw_double(str); 
    
};

template<> void write_raw_vector<double>(ostream& str, const vector<double>& v)
{
  long sz = v.size();  
  write_raw_int(str, sz); 

  for(double n: v)
    write_raw_double(str, n); 
};
