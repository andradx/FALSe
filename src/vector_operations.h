/*
 * vector_operations.h
 *
 *  Created on: Oct 20, 2014
 *      Author: andrade
 */

#ifndef VECTOR_OPERATIONS_H_
#define VECTOR_OPERATIONS_H_


/* int4 vector operations */

__device__ int4 operator +(const int4 &a,const int4 &b){

	return make_int4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
}

__device__ int4 operator *(const int4 &a,const int4 &b){

	return make_int4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);
}

__device__ int4 operator -(const int4 &a,const int4 &b){

	return make_int4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);
}

__device__ int4&  operator +=(int4 &a,const int4 &b){

	a.x+=b.x;
	a.y+=b.y;
	a.z+=b.z;
	a.w+=b.w;

	return a;
}
__device__ int4&  operator -=(int4 &a,const int4 &b){

	a.x-=b.x;
	a.y-=b.y;
	a.z-=b.z;
	a.w-=b.w;

	return a;
}

/* saturated arithmetic on vector types */

__device__ void scalar_clip(int &a,int max_threshold){

	if(a <  - max_threshold)
		a = - max_threshold;
	else if (a > (max_threshold -1))
		a = (max_threshold -1);
}

__device__ void vector_clip(int4 &a,int max_threshold){

	scalar_clip(a.x,max_threshold);
	scalar_clip(a.y,max_threshold);
	scalar_clip(a.z,max_threshold);
	scalar_clip(a.w,max_threshold);
}

/* inter-char4-int4 operations */

__device__ int4 load_char4_to_int4(const char4 &a){

	return make_int4((int)a.x,(int)a.y,(int)a.z,(int)a.w);
}


__device__ char4 store_int4_to_char4(const int4 &a,int max_threshold){

	int4 b = a;
	vector_clip(b,max_threshold);

	return make_char4((char)b.x,(char)b.y,(char)b.z,(char)b.w);
}

/* vector minimum */
__device__ int4 min(const int4 &a,int4 &min1, int4 &min2,int i){

	int4 idx;

	if(a.x < min1.x){
		min1.x = a.x;
		idx.x = i;
	}
	else if(a.x < min2.x)
		min2.x = a.x;

	if(a.y < min1.y){
		min1.y = a.y;
		idx.y = i;
	}
	else if(a.y < min2.y)
		min2.y = a.y;

	if(a.z < min1.z){
		min1.z = a.z;
		idx.z = i;
	}
	else if(a.z < min2.z)
		min2.z = a.z;

	if(a.w < min1.w){
		min1.w = a.w;
		idx.w = i;
	}
	else if(a.w < min2.w)
		min2.w = a.w;

	return make_int4(idx.x,idx.y,idx.z,idx.w);

}
/* scalar minimum */

/* vector absolute */
__device__ int4 abs(const int4 &a){

	return make_int4(abs(a.x),abs(a.y),abs(a.z),abs(a.w));
}

#endif /* VECTOR_OPERATIONS_H_ */
