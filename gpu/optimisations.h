#pragma once

// Optimisation from "GPU implementation of finite difference solvers" (https://people.maths.ox.ac.uk/~gilesm/files/WHPCF14.pdf)
// by Jeremy Appleyard, Nvidia.
// Modernised and simplified by Sacha Refshauge
template <typename REAL>
__forceinline__ __device__
void PCR_2_warp(REAL am, REAL cm, REAL& dm, REAL ap, REAL cp, REAL& dp)
{
	unsigned m = 0xffffffff;
	REAL r;
	int stride = 1;
	r = __rcp(1.0 - ap * cm - cp * __shfl_down_sync(m, am, 1));
	dp = (dp - ap * dm - cp * __shfl_down_sync(m, dm, 1)) * r;
	ap = -ap * am * r;
	cp = -cp * __shfl_down_sync(m, cm, 1) * r;

	for (int n = 0; n < 5; n++) {
		r = __rcp(1.0 - ap * __shfl_up_sync(m, cp, stride) - cp * __shfl_down_sync(m, ap, stride));
		dp = r * (dp - ap * __shfl_up_sync(m, dp, stride) - cp * __shfl_down_sync(m, dp, stride));
		ap = -r * ap * __shfl_up_sync(m, ap, stride);
		cp = -r * cp * __shfl_down_sync(m, cp, stride);
		stride *= 2;
	}

	dm = dm - am * __shfl_up_sync(m, dp, 1) - cm * dp;
}

// While a simple divide works, this helps a lot with performance
// Optimisation from "GPU implementation of finite difference solvers" (https://people.maths.ox.ac.uk/~gilesm/files/WHPCF14.pdf)
// by Mike Giles
static __forceinline__ __device__ float __rcp(float a) {
	return 1.0f / a;
}

static __forceinline__ __device__ double __rcp(double a) {
	double e, y;
	// rcp.approx for double
	asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(y) : "d"(a));
	// __fma_rn(x,y,z)  x*y+z
	// refined to full double precision by extension to Newton iteration
	// a mathmatical method to reduce the relative error
	e = __fma_rn(-a, y, 1.0); //-a*y+1.0
	e = __fma_rn(e, e, e); //-e*e+e
	y = __fma_rn(e, y, y); //-e*y+y
	return y;
}