#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <complex.h>
#ifdef complex
#undef complex
#endif
#ifdef I
#undef I
#endif

#if defined(_WIN64)
typedef long long BLASLONG;
typedef unsigned long long BLASULONG;
#else
typedef long BLASLONG;
typedef unsigned long BLASULONG;
#endif

#ifdef LAPACK_ILP64
typedef BLASLONG blasint;
#if defined(_WIN64)
#define blasabs(x) llabs(x)
#else
#define blasabs(x) labs(x)
#endif
#else
typedef int blasint;
#define blasabs(x) abs(x)
#endif

typedef blasint integer;

typedef unsigned int uinteger;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
#ifdef _MSC_VER
static inline _Fcomplex Cf(complex *z) {_Fcomplex zz={z->r , z->i}; return zz;}
static inline _Dcomplex Cd(doublecomplex *z) {_Dcomplex zz={z->r , z->i};return zz;}
static inline _Fcomplex * _pCf(complex *z) {return (_Fcomplex*)z;}
static inline _Dcomplex * _pCd(doublecomplex *z) {return (_Dcomplex*)z;}
#else
static inline _Complex float Cf(complex *z) {return z->r + z->i*_Complex_I;}
static inline _Complex double Cd(doublecomplex *z) {return z->r + z->i*_Complex_I;}
static inline _Complex float * _pCf(complex *z) {return (_Complex float*)z;}
static inline _Complex double * _pCd(doublecomplex *z) {return (_Complex double*)z;}
#endif
#define pCf(z) (*_pCf(z))
#define pCd(z) (*_pCd(z))
typedef blasint logical;

typedef char logical1;
typedef char integer1;

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

typedef int flag;
typedef int ftnlen;
typedef int ftnint;

/*external read, write*/
typedef struct
{	flag cierr;
	ftnint ciunit;
	flag ciend;
	char *cifmt;
	ftnint cirec;
} cilist;

/*internal read, write*/
typedef struct
{	flag icierr;
	char *iciunit;
	flag iciend;
	char *icifmt;
	ftnint icirlen;
	ftnint icirnum;
} icilist;

/*open*/
typedef struct
{	flag oerr;
	ftnint ounit;
	char *ofnm;
	ftnlen ofnmlen;
	char *osta;
	char *oacc;
	char *ofm;
	ftnint orl;
	char *oblnk;
} olist;

/*close*/
typedef struct
{	flag cerr;
	ftnint cunit;
	char *csta;
} cllist;

/*rewind, backspace, endfile*/
typedef struct
{	flag aerr;
	ftnint aunit;
} alist;

/* inquire */
typedef struct
{	flag inerr;
	ftnint inunit;
	char *infile;
	ftnlen infilen;
	ftnint	*inex;	/*parameters in standard's order*/
	ftnint	*inopen;
	ftnint	*innum;
	ftnint	*innamed;
	char	*inname;
	ftnlen	innamlen;
	char	*inacc;
	ftnlen	inacclen;
	char	*inseq;
	ftnlen	inseqlen;
	char 	*indir;
	ftnlen	indirlen;
	char	*infmt;
	ftnlen	infmtlen;
	char	*inform;
	ftnint	informlen;
	char	*inunf;
	ftnlen	inunflen;
	ftnint	*inrecl;
	ftnint	*innrec;
	char	*inblank;
	ftnlen	inblanklen;
} inlist;

#define VOID void

union Multitype {	/* for multiple entry points */
	integer1 g;
	shortint h;
	integer i;
	/* longint j; */
	real r;
	doublereal d;
	complex c;
	doublecomplex z;
	};

typedef union Multitype Multitype;

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	int  type;
	};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	int nvars;
	};
typedef struct Namelist Namelist;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (fabs(x))
#define f2cmin(a,b) ((a) <= (b) ? (a) : (b))
#define f2cmax(a,b) ((a) >= (b) ? (a) : (b))
#define dmin(a,b) (f2cmin(a,b))
#define dmax(a,b) (f2cmax(a,b))
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((uinteger)1 << (b)))
#define bit_set(a,b)	((a) |  ((uinteger)1 << (b)))

#define abort_() { sig_die("Fortran abort routine called", 1); }
#define c_abs(z) (cabsf(Cf(z)))
#define c_cos(R,Z) { pCf(R)=ccos(Cf(Z)); }
#ifdef _MSC_VER
#define c_div(c, a, b) {Cf(c)._Val[0] = (Cf(a)._Val[0]/Cf(b)._Val[0]); Cf(c)._Val[1]=(Cf(a)._Val[1]/Cf(b)._Val[1]);}
#define z_div(c, a, b) {Cd(c)._Val[0] = (Cd(a)._Val[0]/Cd(b)._Val[0]); Cd(c)._Val[1]=(Cd(a)._Val[1]/df(b)._Val[1]);}
#else
#define c_div(c, a, b) {pCf(c) = Cf(a)/Cf(b);}
#define z_div(c, a, b) {pCd(c) = Cd(a)/Cd(b);}
#endif
#define c_exp(R, Z) {pCf(R) = cexpf(Cf(Z));}
#define c_log(R, Z) {pCf(R) = clogf(Cf(Z));}
#define c_sin(R, Z) {pCf(R) = csinf(Cf(Z));}
//#define c_sqrt(R, Z) {*(R) = csqrtf(Cf(Z));}
#define c_sqrt(R, Z) {pCf(R) = csqrtf(Cf(Z));}
#define d_abs(x) (fabs(*(x)))
#define d_acos(x) (acos(*(x)))
#define d_asin(x) (asin(*(x)))
#define d_atan(x) (atan(*(x)))
#define d_atn2(x, y) (atan2(*(x),*(y)))
#define d_cnjg(R, Z) { pCd(R) = conj(Cd(Z)); }
#define r_cnjg(R, Z) { pCf(R) = conjf(Cf(Z)); }
#define d_cos(x) (cos(*(x)))
#define d_cosh(x) (cosh(*(x)))
#define d_dim(__a, __b) ( *(__a) > *(__b) ? *(__a) - *(__b) : 0.0 )
#define d_exp(x) (exp(*(x)))
#define d_imag(z) (cimag(Cd(z)))
#define r_imag(z) (cimagf(Cf(z)))
#define d_int(__x) (*(__x)>0 ? floor(*(__x)) : -floor(- *(__x)))
#define r_int(__x) (*(__x)>0 ? floor(*(__x)) : -floor(- *(__x)))
#define d_lg10(x) ( 0.43429448190325182765 * log(*(x)) )
#define r_lg10(x) ( 0.43429448190325182765 * log(*(x)) )
#define d_log(x) (log(*(x)))
#define d_mod(x, y) (fmod(*(x), *(y)))
#define u_nint(__x) ((__x)>=0 ? floor((__x) + .5) : -floor(.5 - (__x)))
#define d_nint(x) u_nint(*(x))
#define u_sign(__a,__b) ((__b) >= 0 ? ((__a) >= 0 ? (__a) : -(__a)) : -((__a) >= 0 ? (__a) : -(__a)))
#define d_sign(a,b) u_sign(*(a),*(b))
#define r_sign(a,b) u_sign(*(a),*(b))
#define d_sin(x) (sin(*(x)))
#define d_sinh(x) (sinh(*(x)))
#define d_sqrt(x) (sqrt(*(x)))
#define d_tan(x) (tan(*(x)))
#define d_tanh(x) (tanh(*(x)))
#define i_abs(x) abs(*(x))
#define i_dnnt(x) ((integer)u_nint(*(x)))
#define i_len(s, n) (n)
#define i_nint(x) ((integer)u_nint(*(x)))
#define i_sign(a,b) ((integer)u_sign((integer)*(a),(integer)*(b)))
#define pow_dd(ap, bp) ( pow(*(ap), *(bp)))
#define pow_si(B,E) spow_ui(*(B),*(E))
#define pow_ri(B,E) spow_ui(*(B),*(E))
#define pow_di(B,E) dpow_ui(*(B),*(E))
#define pow_zi(p, a, b) {pCd(p) = zpow_ui(Cd(a), *(b));}
#define pow_ci(p, a, b) {pCf(p) = cpow_ui(Cf(a), *(b));}
#define pow_zz(R,A,B) {pCd(R) = cpow(Cd(A),*(B));}
#define s_cat(lpp, rpp, rnp, np, llp) { 	ftnlen i, nc, ll; char *f__rp, *lp; 	ll = (llp); lp = (lpp); 	for(i=0; i < (int)*(np); ++i) {         	nc = ll; 	        if((rnp)[i] < nc) nc = (rnp)[i]; 	        ll -= nc;         	f__rp = (rpp)[i]; 	        while(--nc >= 0) *lp++ = *(f__rp)++;         } 	while(--ll >= 0) *lp++ = ' '; }
#define s_cmp(a,b,c,d) ((integer)strncmp((a),(b),f2cmin((c),(d))))
#define s_copy(A,B,C,D) { int __i,__m; for (__i=0, __m=f2cmin((C),(D)); __i<__m && (B)[__i] != 0; ++__i) (A)[__i] = (B)[__i]; }
#define sig_die(s, kill) { exit(1); }
#define s_stop(s, n) {exit(0);}
static char junk[] = "\n@(#)LIBF77 VERSION 19990503\n";
#define z_abs(z) (cabs(Cd(z)))
#define z_exp(R, Z) {pCd(R) = cexp(Cd(Z));}
#define z_sqrt(R, Z) {pCd(R) = csqrt(Cd(Z));}
#define myexit_() break;
#define mycycle() continue;
#define myceiling(w) {ceil(w)}
#define myhuge(w) {HUGE_VAL}
//#define mymaxloc_(w,s,e,n) {if (sizeof(*(w)) == sizeof(double)) dmaxloc_((w),*(s),*(e),n); else dmaxloc_((w),*(s),*(e),n);}
#define mymaxloc(w,s,e,n) {dmaxloc_(w,*(s),*(e),n)}

/* procedure parameter types for -A and -C++ */


#ifdef __cplusplus
typedef logical (*L_fp)(...);
#else
typedef logical (*L_fp)();
#endif

static float spow_ui(float x, integer n) {
	float pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
static double dpow_ui(double x, integer n) {
	double pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
#ifdef _MSC_VER
static _Fcomplex cpow_ui(complex x, integer n) {
	complex pow={1.0,0.0}; unsigned long int u;
		if(n != 0) {
		if(n < 0) n = -n, x.r = 1/x.r, x.i=1/x.i;
		for(u = n; ; ) {
			if(u & 01) pow.r *= x.r, pow.i *= x.i;
			if(u >>= 1) x.r *= x.r, x.i *= x.i;
			else break;
		}
	}
	_Fcomplex p={pow.r, pow.i};
	return p;
}
#else
static _Complex float cpow_ui(_Complex float x, integer n) {
	_Complex float pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
#endif
#ifdef _MSC_VER
static _Dcomplex zpow_ui(_Dcomplex x, integer n) {
	_Dcomplex pow={1.0,0.0}; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x._Val[0] = 1/x._Val[0], x._Val[1] =1/x._Val[1];
		for(u = n; ; ) {
			if(u & 01) pow._Val[0] *= x._Val[0], pow._Val[1] *= x._Val[1];
			if(u >>= 1) x._Val[0] *= x._Val[0], x._Val[1] *= x._Val[1];
			else break;
		}
	}
	_Dcomplex p = {pow._Val[0], pow._Val[1]};
	return p;
}
#else
static _Complex double zpow_ui(_Complex double x, integer n) {
	_Complex double pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
#endif
static integer pow_ii(integer x, integer n) {
	integer pow; unsigned long int u;
	if (n <= 0) {
		if (n == 0 || x == 1) pow = 1;
		else if (x != -1) pow = x == 0 ? 1/x : 0;
		else n = -n;
	}
	if ((n > 0) || !(n == 0 || x == 1 || x != -1)) {
		u = n;
		for(pow = 1; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
static integer dmaxloc_(double *w, integer s, integer e, integer *n)
{
	double m; integer i, mi;
	for(m=w[s-1], mi=s, i=s+1; i<=e; i++)
		if (w[i-1]>m) mi=i ,m=w[i-1];
	return mi-s+1;
}
static integer smaxloc_(float *w, integer s, integer e, integer *n)
{
	float m; integer i, mi;
	for(m=w[s-1], mi=s, i=s+1; i<=e; i++)
		if (w[i-1]>m) mi=i ,m=w[i-1];
	return mi-s+1;
}
static inline void cdotc_(complex *z, integer *n_, complex *x, integer *incx_, complex *y, integer *incy_) {
	integer n = *n_, incx = *incx_, incy = *incy_, i;
#ifdef _MSC_VER
	_Fcomplex zdotc = {0.0, 0.0};
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += conjf(Cf(&x[i]))._Val[0] * Cf(&y[i])._Val[0];
			zdotc._Val[1] += conjf(Cf(&x[i]))._Val[1] * Cf(&y[i])._Val[1];
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += conjf(Cf(&x[i*incx]))._Val[0] * Cf(&y[i*incy])._Val[0];
			zdotc._Val[1] += conjf(Cf(&x[i*incx]))._Val[1] * Cf(&y[i*incy])._Val[1];
		}
	}
	pCf(z) = zdotc;
}
#else
	_Complex float zdotc = 0.0;
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += conjf(Cf(&x[i])) * Cf(&y[i]);
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += conjf(Cf(&x[i*incx])) * Cf(&y[i*incy]);
		}
	}
	pCf(z) = zdotc;
}
#endif
static inline void zdotc_(doublecomplex *z, integer *n_, doublecomplex *x, integer *incx_, doublecomplex *y, integer *incy_) {
	integer n = *n_, incx = *incx_, incy = *incy_, i;
#ifdef _MSC_VER
	_Dcomplex zdotc = {0.0, 0.0};
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += conj(Cd(&x[i]))._Val[0] * Cd(&y[i])._Val[0];
			zdotc._Val[1] += conj(Cd(&x[i]))._Val[1] * Cd(&y[i])._Val[1];
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += conj(Cd(&x[i*incx]))._Val[0] * Cd(&y[i*incy])._Val[0];
			zdotc._Val[1] += conj(Cd(&x[i*incx]))._Val[1] * Cd(&y[i*incy])._Val[1];
		}
	}
	pCd(z) = zdotc;
}
#else
	_Complex double zdotc = 0.0;
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += conj(Cd(&x[i])) * Cd(&y[i]);
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += conj(Cd(&x[i*incx])) * Cd(&y[i*incy]);
		}
	}
	pCd(z) = zdotc;
}
#endif	
static inline void cdotu_(complex *z, integer *n_, complex *x, integer *incx_, complex *y, integer *incy_) {
	integer n = *n_, incx = *incx_, incy = *incy_, i;
#ifdef _MSC_VER
	_Fcomplex zdotc = {0.0, 0.0};
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += Cf(&x[i])._Val[0] * Cf(&y[i])._Val[0];
			zdotc._Val[1] += Cf(&x[i])._Val[1] * Cf(&y[i])._Val[1];
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += Cf(&x[i*incx])._Val[0] * Cf(&y[i*incy])._Val[0];
			zdotc._Val[1] += Cf(&x[i*incx])._Val[1] * Cf(&y[i*incy])._Val[1];
		}
	}
	pCf(z) = zdotc;
}
#else
	_Complex float zdotc = 0.0;
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += Cf(&x[i]) * Cf(&y[i]);
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += Cf(&x[i*incx]) * Cf(&y[i*incy]);
		}
	}
	pCf(z) = zdotc;
}
#endif
static inline void zdotu_(doublecomplex *z, integer *n_, doublecomplex *x, integer *incx_, doublecomplex *y, integer *incy_) {
	integer n = *n_, incx = *incx_, incy = *incy_, i;
#ifdef _MSC_VER
	_Dcomplex zdotc = {0.0, 0.0};
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += Cd(&x[i])._Val[0] * Cd(&y[i])._Val[0];
			zdotc._Val[1] += Cd(&x[i])._Val[1] * Cd(&y[i])._Val[1];
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc._Val[0] += Cd(&x[i*incx])._Val[0] * Cd(&y[i*incy])._Val[0];
			zdotc._Val[1] += Cd(&x[i*incx])._Val[1] * Cd(&y[i*incy])._Val[1];
		}
	}
	pCd(z) = zdotc;
}
#else
	_Complex double zdotc = 0.0;
	if (incx == 1 && incy == 1) {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += Cd(&x[i]) * Cd(&y[i]);
		}
	} else {
		for (i=0;i<n;i++) { /* zdotc = zdotc + dconjg(x(i))* y(i) */
			zdotc += Cd(&x[i*incx]) * Cd(&y[i*incy]);
		}
	}
	pCd(z) = zdotc;
}
#endif
/*  -- translated by f2c (version 20000121).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/




/* > \brief \b DTPTTF copies a triangular matrix from the standard packed format (TP) to the rectangular full 
packed format (TF). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DTPTTF + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dtpttf.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dtpttf.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dtpttf.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DTPTTF( TRANSR, UPLO, N, AP, ARF, INFO ) */

/*       CHARACTER          TRANSR, UPLO */
/*       INTEGER            INFO, N */
/*       DOUBLE PRECISION   AP( 0: * ), ARF( 0: * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DTPTTF copies a triangular matrix A from standard packed format (TP) */
/* > to rectangular full packed format (TF). */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] TRANSR */
/* > \verbatim */
/* >          TRANSR is CHARACTER*1 */
/* >          = 'N':  ARF in Normal format is wanted; */
/* >          = 'T':  ARF in Conjugate-transpose format is wanted. */
/* > \endverbatim */
/* > */
/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  A is upper triangular; */
/* >          = 'L':  A is lower triangular. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] AP */
/* > \verbatim */
/* >          AP is DOUBLE PRECISION array, dimension ( N*(N+1)/2 ), */
/* >          On entry, the upper or lower triangular matrix A, packed */
/* >          columnwise in a linear array. The j-th column of A is stored */
/* >          in the array AP as follows: */
/* >          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/* >          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */
/* > \endverbatim */
/* > */
/* > \param[out] ARF */
/* > \verbatim */
/* >          ARF is DOUBLE PRECISION array, dimension ( N*(N+1)/2 ), */
/* >          On exit, the upper or lower triangular matrix A stored in */
/* >          RFP format. For a further discussion see Notes below. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doubleOTHERcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  We first consider Rectangular Full Packed (RFP) Format when N is */
/* >  even. We give an example where N = 6. */
/* > */
/* >      AP is Upper             AP is Lower */
/* > */
/* >   00 01 02 03 04 05       00 */
/* >      11 12 13 14 15       10 11 */
/* >         22 23 24 25       20 21 22 */
/* >            33 34 35       30 31 32 33 */
/* >               44 45       40 41 42 43 44 */
/* >                  55       50 51 52 53 54 55 */
/* > */
/* > */
/* >  Let TRANSR = 'N'. RFP holds AP as follows: */
/* >  For UPLO = 'U' the upper trapezoid A(0:5,0:2) consists of the last */
/* >  three columns of AP upper. The lower triangle A(4:6,0:2) consists of */
/* >  the transpose of the first three columns of AP upper. */
/* >  For UPLO = 'L' the lower trapezoid A(1:6,0:2) consists of the first */
/* >  three columns of AP lower. The upper triangle A(0:2,0:2) consists of */
/* >  the transpose of the last three columns of AP lower. */
/* >  This covers the case N even and TRANSR = 'N'. */
/* > */
/* >         RFP A                   RFP A */
/* > */
/* >        03 04 05                33 43 53 */
/* >        13 14 15                00 44 54 */
/* >        23 24 25                10 11 55 */
/* >        33 34 35                20 21 22 */
/* >        00 44 45                30 31 32 */
/* >        01 11 55                40 41 42 */
/* >        02 12 22                50 51 52 */
/* > */
/* >  Now let TRANSR = 'T'. RFP A in both UPLO cases is just the */
/* >  transpose of RFP A above. One therefore gets: */
/* > */
/* > */
/* >           RFP A                   RFP A */
/* > */
/* >     03 13 23 33 00 01 02    33 00 10 20 30 40 50 */
/* >     04 14 24 34 44 11 12    43 44 11 21 31 41 51 */
/* >     05 15 25 35 45 55 22    53 54 55 22 32 42 52 */
/* > */
/* > */
/* >  We then consider Rectangular Full Packed (RFP) Format when N is */
/* >  odd. We give an example where N = 5. */
/* > */
/* >     AP is Upper                 AP is Lower */
/* > */
/* >   00 01 02 03 04              00 */
/* >      11 12 13 14              10 11 */
/* >         22 23 24              20 21 22 */
/* >            33 34              30 31 32 33 */
/* >               44              40 41 42 43 44 */
/* > */
/* > */
/* >  Let TRANSR = 'N'. RFP holds AP as follows: */
/* >  For UPLO = 'U' the upper trapezoid A(0:4,0:2) consists of the last */
/* >  three columns of AP upper. The lower triangle A(3:4,0:1) consists of */
/* >  the transpose of the first two columns of AP upper. */
/* >  For UPLO = 'L' the lower trapezoid A(0:4,0:2) consists of the first */
/* >  three columns of AP lower. The upper triangle A(0:1,1:2) consists of */
/* >  the transpose of the last two columns of AP lower. */
/* >  This covers the case N odd and TRANSR = 'N'. */
/* > */
/* >         RFP A                   RFP A */
/* > */
/* >        02 03 04                00 33 43 */
/* >        12 13 14                10 11 44 */
/* >        22 23 24                20 21 22 */
/* >        00 33 34                30 31 32 */
/* >        01 11 44                40 41 42 */
/* > */
/* >  Now let TRANSR = 'T'. RFP A in both UPLO cases is just the */
/* >  transpose of RFP A above. One therefore gets: */
/* > */
/* >           RFP A                   RFP A */
/* > */
/* >     02 12 22 00 01             00 10 20 30 40 50 */
/* >     03 13 23 33 11             33 11 21 31 41 51 */
/* >     04 14 24 34 44             43 44 22 32 42 52 */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
/* Subroutine */ void dtpttf_(char *transr, char *uplo, integer *n, doublereal 
	*ap, doublereal *arf, integer *info)
{
    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Local variables */
    integer i__, j, k;
    logical normaltransr;
    extern logical lsame_(char *, char *);
    logical lower;
    integer n1, n2, ij, jp, js, nt;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    logical nisodd;
    integer lda, ijp;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    *info = 0;
    normaltransr = lsame_(transr, "N");
    lower = lsame_(uplo, "L");
    if (! normaltransr && ! lsame_(transr, "T")) {
	*info = -1;
    } else if (! lower && ! lsame_(uplo, "U")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DTPTTF", &i__1, (ftnlen)6);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

    if (*n == 1) {
	if (normaltransr) {
	    arf[0] = ap[0];
	} else {
	    arf[0] = ap[0];
	}
	return;
    }

/*     Size of array ARF(0:NT-1) */

    nt = *n * (*n + 1) / 2;

/*     Set N1 and N2 depending on LOWER */

    if (lower) {
	n2 = *n / 2;
	n1 = *n - n2;
    } else {
	n1 = *n / 2;
	n2 = *n - n1;
    }

/*     If N is odd, set NISODD = .TRUE. */
/*     If N is even, set K = N/2 and NISODD = .FALSE. */

/*     set lda of ARF^C; ARF^C is (0:(N+1)/2-1,0:N-noe) */
/*     where noe = 0 if n is even, noe = 1 if n is odd */

    if (*n % 2 == 0) {
	k = *n / 2;
	nisodd = FALSE_;
	lda = *n + 1;
    } else {
	nisodd = TRUE_;
	lda = *n;
    }

/*     ARF^C has lda rows and n+1-noe cols */

    if (! normaltransr) {
	lda = (*n + 1) / 2;
    }

/*     start execution: there are eight cases */

    if (nisodd) {

/*        N is odd */

	if (normaltransr) {

/*           N is odd and TRANSR = 'N' */

	    if (lower) {

/*              N is odd, TRANSR = 'N', and UPLO = 'L' */

		ijp = 0;
		jp = 0;
		i__1 = n2;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = *n - 1;
		    for (i__ = j; i__ <= i__2; ++i__) {
			ij = i__ + jp;
			arf[ij] = ap[ijp];
			++ijp;
		    }
		    jp += lda;
		}
		i__1 = n2 - 1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__2 = n2;
		    for (j = i__ + 1; j <= i__2; ++j) {
			ij = i__ + j * lda;
			arf[ij] = ap[ijp];
			++ijp;
		    }
		}

	    } else {

/*              N is odd, TRANSR = 'N', and UPLO = 'U' */

		ijp = 0;
		i__1 = n1 - 1;
		for (j = 0; j <= i__1; ++j) {
		    ij = n2 + j;
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			arf[ij] = ap[ijp];
			++ijp;
			ij += lda;
		    }
		}
		js = 0;
		i__1 = *n - 1;
		for (j = n1; j <= i__1; ++j) {
		    ij = js;
		    i__2 = js + j;
		    for (ij = js; ij <= i__2; ++ij) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		    js += lda;
		}

	    }

	} else {

/*           N is odd and TRANSR = 'T' */

	    if (lower) {

/*              N is odd, TRANSR = 'T', and UPLO = 'L' */

		ijp = 0;
		i__1 = n2;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__2 = *n * lda - 1;
		    i__3 = lda;
		    for (ij = i__ * (lda + 1); i__3 < 0 ? ij >= i__2 : ij <= 
			    i__2; ij += i__3) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		}
		js = 1;
		i__1 = n2 - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__3 = js + n2 - j - 1;
		    for (ij = js; ij <= i__3; ++ij) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		    js = js + lda + 1;
		}

	    } else {

/*              N is odd, TRANSR = 'T', and UPLO = 'U' */

		ijp = 0;
		js = n2 * lda;
		i__1 = n1 - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__3 = js + j;
		    for (ij = js; ij <= i__3; ++ij) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		    js += lda;
		}
		i__1 = n1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__3 = i__ + (n1 + i__) * lda;
		    i__2 = lda;
		    for (ij = i__; i__2 < 0 ? ij >= i__3 : ij <= i__3; ij += 
			    i__2) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		}

	    }

	}

    } else {

/*        N is even */

	if (normaltransr) {

/*           N is even and TRANSR = 'N' */

	    if (lower) {

/*              N is even, TRANSR = 'N', and UPLO = 'L' */

		ijp = 0;
		jp = 0;
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = *n - 1;
		    for (i__ = j; i__ <= i__2; ++i__) {
			ij = i__ + 1 + jp;
			arf[ij] = ap[ijp];
			++ijp;
		    }
		    jp += lda;
		}
		i__1 = k - 1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__2 = k - 1;
		    for (j = i__; j <= i__2; ++j) {
			ij = i__ + j * lda;
			arf[ij] = ap[ijp];
			++ijp;
		    }
		}

	    } else {

/*              N is even, TRANSR = 'N', and UPLO = 'U' */

		ijp = 0;
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    ij = k + 1 + j;
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			arf[ij] = ap[ijp];
			++ijp;
			ij += lda;
		    }
		}
		js = 0;
		i__1 = *n - 1;
		for (j = k; j <= i__1; ++j) {
		    ij = js;
		    i__2 = js + j;
		    for (ij = js; ij <= i__2; ++ij) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		    js += lda;
		}

	    }

	} else {

/*           N is even and TRANSR = 'T' */

	    if (lower) {

/*              N is even, TRANSR = 'T', and UPLO = 'L' */

		ijp = 0;
		i__1 = k - 1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__2 = (*n + 1) * lda - 1;
		    i__3 = lda;
		    for (ij = i__ + (i__ + 1) * lda; i__3 < 0 ? ij >= i__2 : 
			    ij <= i__2; ij += i__3) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		}
		js = 0;
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__3 = js + k - j - 1;
		    for (ij = js; ij <= i__3; ++ij) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		    js = js + lda + 1;
		}

	    } else {

/*              N is even, TRANSR = 'T', and UPLO = 'U' */

		ijp = 0;
		js = (k + 1) * lda;
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__3 = js + j;
		    for (ij = js; ij <= i__3; ++ij) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		    js += lda;
		}
		i__1 = k - 1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__3 = i__ + (k + i__) * lda;
		    i__2 = lda;
		    for (ij = i__; i__2 < 0 ? ij >= i__3 : ij <= i__3; ij += 
			    i__2) {
			arf[ij] = ap[ijp];
			++ijp;
		    }
		}

	    }

	}

    }

    return;

/*     End of DTPTTF */

} /* dtpttf_ */

