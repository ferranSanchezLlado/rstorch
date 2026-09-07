#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// CUDA counterpart of the scalar Metal kernels. All scalar launch parameters
// are passed by value; shape and stride arrays remain device pointers.
using u64 = uint64_t;
using i64 = int64_t;
using u32 = uint32_t;
using u8 = uint8_t;

static __device__ __forceinline__ u64 global_id() {
    return (u64)blockIdx.x * blockDim.x + threadIdx.x;
}

static __device__ __forceinline__ u64 view_offset(
    u64 logical, const u64* dims, const u64* strides, u32 rank, u64 offset
) {
    for (u32 axis = rank; axis > 0; --axis) {
        u64 dim = dims[axis - 1];
        if (dim != 0) {
            offset += (logical % dim) * strides[axis - 1];
            logical /= dim;
        }
    }
    return offset;
}

static __device__ __forceinline__ u64 logical_coord(
    u64 logical, const u64* dims, u32 rank, u32 wanted
) {
    for (u32 axis = rank; axis > 0; --axis) {
        u64 dim = dims[axis - 1];
        u64 coord = dim == 0 ? 0 : logical % dim;
        if (axis - 1 == wanted) return coord;
        if (dim != 0) logical /= dim;
    }
    return 0;
}

static __device__ __forceinline__ u64 reduced_offset(
    u64 outer, const u64* dims, const u64* strides,
    u32 rank, u32 reduced_axis, u64 offset
) {
    for (u32 axis = rank; axis > 0; --axis) {
        u32 a = axis - 1;
        if (a == reduced_axis) continue;
        u64 dim = dims[a];
        if (dim != 0) {
            offset += (outer % dim) * strides[a];
            outer /= dim;
        }
    }
    return offset;
}

static __device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
static __device__ __forceinline__ float to_f32(float v) { return v; }
static __device__ __forceinline__ i64 to_i64(i64 v) { return v; }
static __device__ __forceinline__ __half from_f16(float v) { return __float2half_rn(v); }
static __device__ __forceinline__ float from_f32(float v) { return v; }
static __device__ __forceinline__ i64 from_i64(i64 v) { return v; }
static __device__ __forceinline__ bool value_nan(float v) { return isnan(v); }
static __device__ __forceinline__ bool value_nan(i64) { return false; }

static __device__ __forceinline__ i64 wrap_add(i64 a, i64 b) {
    return (i64)((u64)a + (u64)b);
}
static __device__ __forceinline__ i64 wrap_sub(i64 a, i64 b) {
    return (i64)((u64)a - (u64)b);
}
static __device__ __forceinline__ i64 wrap_mul(i64 a, i64 b) {
    return (i64)((u64)a * (u64)b);
}
static __device__ __forceinline__ i64 safe_div(i64 a, i64 b) {
    const i64 low = (i64)(UINT64_C(1) << 63);
    return b == 0 ? 0 : (a == low && b == -1 ? low : a / b);
}

#define COPY_KERNEL(TYPE, NAME) \
extern "C" __global__ void copy_##NAME(const TYPE* src, TYPE* dst, const u64* dims, const u64* strides, u32 rank, u64 offset, u64 len, u32 contiguous) { \
    u64 gid=global_id(); if(gid<len) dst[gid]=src[contiguous?offset+gid:view_offset(gid,dims,strides,rank,offset)]; \
} \
extern "C" __global__ void copy_into_##NAME(const TYPE* src, TYPE* dst, const u64* sd, const u64* ss, u32 sr, u64 so, const u64* dd, const u64* ds, u32 dr, u64 dso, u64 len, u32 contiguous) { \
    u64 gid=global_id(); if(gid<len){u64 from=contiguous?so+gid:view_offset(gid,sd,ss,sr,so);u64 to=contiguous?dso+gid:view_offset(gid,dd,ds,dr,dso);dst[to]=src[from];} \
}
COPY_KERNEL(__half, f16)
COPY_KERNEL(float, f32)
COPY_KERNEL(i64, i64)
COPY_KERNEL(u8, bool)

extern "C" __global__ void full_f16(__half* out, u64 len, float value) {
    u64 gid = global_id();
    if (gid < len) out[gid] = from_f16(value);
}
extern "C" __global__ void full_f32(float* out, u64 len, float value) {
    u64 gid = global_id();
    if (gid < len) out[gid] = value;
}
extern "C" __global__ void full_i64(i64* out, u64 len, i64 value) {
    u64 gid = global_id();
    if (gid < len) out[gid] = value;
}
extern "C" __global__ void full_bool(u8* out, u64 len, u32 value) {
    u64 gid = global_id();
    if (gid < len) out[gid] = (u8)value;
}

extern "C" __global__ void validate_indices(
    const i64* idx, i64* result, const u64* dims, const u64* strides,
    u32 rank, u64 offset, u64 len, u64 bound, u64 slot
) {
    if (global_id() != 0) return;
    result += slot * 2;
    result[0]=0; result[1]=0;
    for(u64 i=0;i<len;++i){i64 v=idx[view_offset(i,dims,strides,rank,offset)];if(v<0||(u64)v>=bound){result[0]=1;result[1]=v;return;}}
}

#define FLOAT_ADD(a,b) ((a)+(b))
#define FLOAT_SUB(a,b) ((a)-(b))
#define FLOAT_MUL(a,b) ((a)*(b))
#define FLOAT_DIV(a,b) ((a)/(b))
#define INT_ADD(a,b) wrap_add((a),(b))
#define INT_SUB(a,b) wrap_sub((a),(b))
#define INT_MUL(a,b) wrap_mul((a),(b))
#define INT_DIV(a,b) safe_div((a),(b))

#define NUMERIC_KERNELS(TYPE, ACC, NAME, TO, FROM, ADD, SUB, MUL, DIV) \
extern "C" __global__ void binary_##NAME(const TYPE* lhs,const TYPE* rhs,TYPE* out,const u64* ld,const u64* ls,u32 lr,u64 lo,const u64* rd,const u64* rs,u32 rr,u64 ro,u64 len,u32 op,u32 contiguous){ \
 u64 gid=global_id();if(gid>=len)return;ACC a=TO(lhs[contiguous?gid:view_offset(gid,ld,ls,lr,lo)]),b=TO(rhs[contiguous?gid:view_offset(gid,rd,rs,rr,ro)]),v; \
 switch(op){case 0:v=ADD(a,b);break;case 1:v=SUB(a,b);break;case 2:v=MUL(a,b);break;case 3:v=DIV(a,b);break;case 4:v=value_nan(a)?a:(value_nan(b)?b:(a>=b?a:b));break;default:v=value_nan(a)?a:(value_nan(b)?b:(a<=b?a:b));}out[gid]=FROM(v); \
} \
extern "C" __global__ void scalar_##NAME(const TYPE* x,TYPE* out,const u64* d,const u64* s,u32 r,u64 off,u64 len,ACC scalar,u32 op,u32 contiguous){ \
 u64 gid=global_id();if(gid>=len)return;ACC a=TO(x[contiguous?gid:view_offset(gid,d,s,r,off)]),b=scalar,v;switch(op){case 0:v=ADD(a,b);break;case 1:v=SUB(a,b);break;case 2:v=MUL(a,b);break;case 3:v=DIV(a,b);break;case 4:v=value_nan(a)?a:(value_nan(b)?b:(a>=b?a:b));break;default:v=value_nan(a)?a:(value_nan(b)?b:(a<=b?a:b));}out[gid]=FROM(v); \
} \
extern "C" __global__ void compare_##NAME(const TYPE* lhs,const TYPE* rhs,u8* out,const u64* ld,const u64* ls,u32 lr,u64 lo,const u64* rd,const u64* rs,u32 rr,u64 ro,u64 len,u32 op,u32 contiguous){ \
 u64 gid=global_id();if(gid>=len)return;ACC a=TO(lhs[contiguous?gid:view_offset(gid,ld,ls,lr,lo)]),b=TO(rhs[contiguous?gid:view_offset(gid,rd,rs,rr,ro)]);bool v;switch(op){case 0:v=a==b;break;case 1:v=a!=b;break;case 2:v=a<b;break;case 3:v=a<=b;break;case 4:v=a>b;break;default:v=a>=b;}out[gid]=(u8)v; \
} \
extern "C" __global__ void reduce_##NAME(const TYPE* x,TYPE* out,const u64* d,const u64* s,u32 r,u64 off,u64 out_len,u32 axis,u32 op){ \
 u64 gid=global_id();if(gid>=out_len)return;u64 base=reduced_offset(gid,d,s,r,axis,off),n=d[axis];ACC acc=(ACC)0;for(u64 i=0;i<n;i++){ACC v=TO(x[base+i*s[axis]]);if(op<2)acc=ADD(acc,v);else if(i==0)acc=v;else if(value_nan(acc)||value_nan(v))acc=ADD(acc,v);else if(op==2)acc=acc>=v?acc:v;else acc=acc<=v?acc:v;}if(op==1)acc=DIV(acc,(ACC)n);out[gid]=FROM(acc); \
} \
extern "C" __global__ void arg_reduce_##NAME(const TYPE* x,i64* out,const u64* d,const u64* s,u32 r,u64 off,u64 out_len,u32 axis,u32 op){ \
 u64 gid=global_id();if(gid>=out_len)return;u64 base=reduced_offset(gid,d,s,r,axis,off),n=d[axis],best=0;ACC bv=TO(x[base]);for(u64 i=1;i<n;i++){ACC v=TO(x[base+i*s[axis]]);bool better=value_nan(bv)?false:(value_nan(v)?true:(op==0?v>bv:v<bv));if(better){bv=v;best=i;}}out[gid]=(i64)best; \
} \
extern "C" __global__ void matmul_##NAME(const TYPE* lhs,const TYPE* rhs,TYPE* out,const u64* bd,const u64* lbs,const u64* rbs,u32 br,const u64* p,u64 len){ \
 u64 gid=global_id();if(gid>=len)return;u64 m=p[0],k=p[1],n=p[2],li=p[3],ri=p[4],batch=gid/(m*n),within=gid%(m*n),b=batch;for(u32 a=br;a>0;--a){u64 dim=bd[a-1],c=dim==0?0:b%dim;if(dim)b/=dim;li+=c*lbs[a-1];ri+=c*rbs[a-1];}u64 row=within/n,col=within%n;ACC acc=(ACC)0;for(u64 q=0;q<k;q++)acc=ADD(acc,MUL(TO(lhs[li+row*p[5]+q*p[6]]),TO(rhs[ri+q*p[7]+col*p[8]])));out[gid]=FROM(acc); \
}
NUMERIC_KERNELS(__half,float,f16,to_f32,from_f16,FLOAT_ADD,FLOAT_SUB,FLOAT_MUL,FLOAT_DIV)
NUMERIC_KERNELS(float,float,f32,to_f32,from_f32,FLOAT_ADD,FLOAT_SUB,FLOAT_MUL,FLOAT_DIV)
NUMERIC_KERNELS(i64,i64,i64,to_i64,from_i64,INT_ADD,INT_SUB,INT_MUL,INT_DIV)

#define MATMUL_TILE 16
#define TILED_MATMUL(TYPE, NAME, TO, FROM) \
extern "C" __global__ void matmul_tiled_##NAME(const TYPE* lhs,const TYPE* rhs,TYPE* out,const u64* bd,const u64* lbs,const u64* rbs,u32 br,const u64* p,u64 len){ \
 __shared__ float lt[MATMUL_TILE][MATMUL_TILE];__shared__ float rt[MATMUL_TILE][MATMUL_TILE]; \
 u64 m=p[0],k=p[1],n=p[2],batch=blockIdx.z,row=(u64)blockIdx.y*MATMUL_TILE+threadIdx.y,col=(u64)blockIdx.x*MATMUL_TILE+threadIdx.x,li=p[3],ri=p[4],b=batch; \
 for(u32 a=br;a>0;--a){u64 dim=bd[a-1],c=dim==0?0:b%dim;if(dim)b/=dim;li+=c*lbs[a-1];ri+=c*rbs[a-1];} \
 float acc=0.0f;for(u64 tile=0;tile<k;tile+=MATMUL_TILE){u64 lq=tile+threadIdx.x,rq=tile+threadIdx.y;lt[threadIdx.y][threadIdx.x]=(row<m&&lq<k)?TO(lhs[li+row*p[5]+lq*p[6]]):0.0f;rt[threadIdx.y][threadIdx.x]=(rq<k&&col<n)?TO(rhs[ri+rq*p[7]+col*p[8]]):0.0f;__syncthreads();for(u32 q=0;q<MATMUL_TILE;q++)acc+=lt[threadIdx.y][q]*rt[q][threadIdx.x];__syncthreads();} \
 u64 index=(batch*m+row)*n+col;if(row<m&&col<n&&index<len)out[index]=FROM(acc); \
}
TILED_MATMUL(__half,f16,to_f32,from_f16)
TILED_MATMUL(float,f32,to_f32,from_f32)

extern "C" __global__ void compare_bool(const u8* lhs,const u8* rhs,u8* out,const u64* ld,const u64* ls,u32 lr,u64 lo,const u64* rd,const u64* rs,u32 rr,u64 ro,u64 len,u32 op,u32 contiguous){
 u64 gid=global_id();if(gid>=len)return;u8 a=lhs[contiguous?gid:view_offset(gid,ld,ls,lr,lo)],b=rhs[contiguous?gid:view_offset(gid,rd,rs,rr,ro)];bool v;switch(op){case 0:v=a==b;break;case 1:v=a!=b;break;case 2:v=a<b;break;case 3:v=a<=b;break;case 4:v=a>b;break;default:v=a>=b;}out[gid]=(u8)v;
}

static __device__ __forceinline__ float erf_accurate(float x) {
    float ax=fabsf(x),value;
    if(ax<0.84375f){float z=ax*ax;float r=0.1283791671f+z*(-0.3250421073f+z*(-0.0284817498f+z*(-0.0057702702f+z*-0.0000237630f)));float s=1.0f+z*(0.3979172111f+z*(0.0650222525f+z*(0.0050813062f+z*(0.0001324947f+z*-0.0000039602f))));value=ax+ax*(r/s);}
    else if(ax<1.25f){float z=ax-1.0f;float p=-0.0023621186f+z*(0.4148561060f+z*(-0.3722078800f+z*(0.3183466196f+z*(-0.1108946949f+z*(0.0354783051f+z*-0.0021663755f)))));float q=1.0f+z*(0.1064208820f+z*(0.5403979421f+z*(0.0718286559f+z*(0.1261712164f+z*(0.0136370836f+z*0.0119845001f)))));value=0.8450629115f+p/q;}
    else{float t=1.0f/(1.0f+0.5f*ax);float tau=t*expf(-ax*ax-1.26551223f+t*(1.00002368f+t*(0.37409196f+t*(0.09678418f+t*(-0.18628806f+t*(0.27886807f+t*(-1.13520398f+t*(1.48851587f+t*(-0.82215223f+t*0.17087277f)))))))));value=1.0f-tau;}
    return x<0.0f?-value:value;
}

#define FLOAT_UNARY(TYPE,NAME,TO,FROM) \
extern "C" __global__ void unary_##NAME(const TYPE* x,TYPE* out,const u64* d,const u64* s,u32 r,u64 off,u64 len,u32 op,u32 contiguous){u64 gid=global_id();if(gid>=len)return;float v=TO(x[contiguous?gid:view_offset(gid,d,s,r,off)]),z;switch(op){case 0:z=isnan(v)?v:fmaxf(v,0.0f);break;case 1:z=0.5f*v*(1.0f+erf_accurate(v*0.7071067811865475f));break;case 2:z=expf(v);break;case 3:z=logf(v);break;case 4:z=sqrtf(v);break;case 5:{float a=fabsf(v);z=copysignf(a>=10.0f?1.0f:tanhf(a),v);}break;case 6:z=1.0f/(1.0f+expf(-v));break;case 7:z=-v;break;default:z=fabsf(v);}out[gid]=FROM(z);}
FLOAT_UNARY(__half,f16,to_f32,from_f16)
FLOAT_UNARY(float,f32,to_f32,from_f32)

extern "C" __global__ void unary_i64(const i64* x,i64* out,const u64* d,const u64* s,u32 r,u64 off,u64 len,u32 op,u32 contiguous){
 u64 gid=global_id();if(gid>=len)return;i64 v=x[contiguous?gid:view_offset(gid,d,s,r,off)],neg=(i64)(u64(0)-(u64)v);out[gid]=op==7?neg:(v<0?neg:v);
}

#define SELECT_KERNELS(TYPE,NAME,FROM,FILL) \
extern "C" __global__ void where_##NAME(const u8* c,const TYPE* t,const TYPE* f,TYPE* out,const u64* cd,const u64* cs,u32 cr,u64 co,const u64* td,const u64* ts,u32 tr,u64 to,const u64* fd,const u64* fs,u32 fr,u64 fo,u64 len,u32 contiguous){u64 gid=global_id();if(gid<len)out[gid]=c[contiguous?gid:view_offset(gid,cd,cs,cr,co)]?t[contiguous?gid:view_offset(gid,td,ts,tr,to)]:f[contiguous?gid:view_offset(gid,fd,fs,fr,fo)];} \
extern "C" __global__ void masked_##NAME(const TYPE* x,const u8* m,TYPE* out,const u64* xd,const u64* xs,u32 xr,u64 xo,const u64* md,const u64* ms,u32 mr,u64 mo,u64 len,FILL fill,u32 contiguous){u64 gid=global_id();if(gid<len)out[gid]=m[contiguous?gid:view_offset(gid,md,ms,mr,mo)]?FROM(fill):x[contiguous?gid:view_offset(gid,xd,xs,xr,xo)];}
SELECT_KERNELS(__half,f16,from_f16,float)
SELECT_KERNELS(float,f32,from_f32,float)
SELECT_KERNELS(i64,i64,from_i64,i64)
static __device__ __forceinline__ u8 from_bool_fill(float v){return (u8)v;}
SELECT_KERNELS(u8,bool,from_bool_fill,float)

static __device__ __forceinline__ i64 float_to_i64(float v){
 if(isnan(v))return 0;if(v>=0x1p63f)return INT64_MAX;if(v<=-0x1p63f)return INT64_MIN;return (i64)v;
}
#define CAST_KERNEL(FROM,TO,FN,TN,EXPR) \
extern "C" __global__ void cast_##FN##_to_##TN(const FROM* x,TO* out,const u64* d,const u64* s,u32 r,u64 off,u64 len,u32 contiguous){u64 gid=global_id();if(gid<len){FROM v=x[contiguous?gid:view_offset(gid,d,s,r,off)];out[gid]=(EXPR);}}
CAST_KERNEL(__half,float,f16,f32,to_f32(v))
CAST_KERNEL(float,__half,f32,f16,from_f16(v))
CAST_KERNEL(__half,i64,f16,i64,float_to_i64(to_f32(v)))
CAST_KERNEL(i64,__half,i64,f16,from_f16((float)v))
CAST_KERNEL(float,i64,f32,i64,float_to_i64(v))
CAST_KERNEL(i64,float,i64,f32,(float)v)
CAST_KERNEL(__half,u8,f16,bool,(u8)(to_f32(v)!=0.0f))
CAST_KERNEL(u8,__half,bool,f16,from_f16(v!=0?1.0f:0.0f))
CAST_KERNEL(float,u8,f32,bool,(u8)(v!=0.0f))
CAST_KERNEL(u8,float,bool,f32,(float)(v!=0))
CAST_KERNEL(i64,u8,i64,bool,(u8)(v!=0))
CAST_KERNEL(u8,i64,bool,i64,(i64)(v!=0))

#define INDEX_KERNELS(TYPE,ACC,NAME,TO,FROM,ADD) \
extern "C" __global__ void index_select_##NAME(const TYPE* x,const i64* idx,TYPE* out,const u64* xd,const u64* xs,u32 xr,u64 xo,const u64* id,const u64* is,u32 ir,u64 io,const u64* od,u32 axis,u64 len){u64 gid=global_id();if(gid>=len)return;i64 raw=idx[view_offset(logical_coord(gid,od,xr,axis),id,is,ir,io)];if(raw<0||(u64)raw>=xd[axis]){out[gid]=FROM((ACC)0);return;}u64 base=xo;for(u32 a=0;a<xr;a++){u64 c=logical_coord(gid,od,xr,a);base+=(a==axis?(u64)raw:c)*xs[a];}out[gid]=x[base];} \
extern "C" __global__ void gather_##NAME(const TYPE* x,const i64* idx,TYPE* out,const u64* xd,const u64* xs,u32 xr,u64 xo,const u64* id,const u64* is,u32 ir,u64 io,u32 axis,u64 len){u64 gid=global_id();if(gid>=len)return;i64 raw=idx[view_offset(gid,id,is,ir,io)];if(raw<0||(u64)raw>=xd[axis]){out[gid]=FROM((ACC)0);return;}u64 base=xo;for(u32 a=0;a<xr;a++){u64 c=logical_coord(gid,id,ir,a);base+=(a==axis?(u64)raw:c)*xs[a];}out[gid]=x[base];} \
extern "C" __global__ void index_add_##NAME(const TYPE* x,const i64* idx,const TYPE* src,TYPE* out,const u64* xd,const u64* xs,u32 xr,u64 xo,const u64* id,const u64* is,u32 ir,u64 io,const u64* sd,const u64* ss,u32 sr,u64 so,u32 axis,u64 out_len,u64 src_len){u64 gid=global_id();if(gid>=out_len)return;ACC acc=TO(x[view_offset(gid,xd,xs,xr,xo)]);for(u64 q=0;q<src_len;q++){bool same=true;for(u32 a=0;a<xr;a++){u64 dc=logical_coord(gid,xd,xr,a),sc=logical_coord(q,sd,sr,a);if(a==axis){i64 raw=idx[view_offset(sc,id,is,ir,io)];if(raw<0||(u64)raw>=xd[axis]){same=false;break;}sc=(u64)raw;}if(dc!=sc)same=false;}if(same)acc=ADD(acc,TO(src[view_offset(q,sd,ss,sr,so)]));}out[gid]=FROM(acc);} \
extern "C" __global__ void scatter_add_##NAME(const TYPE* x,const i64* idx,const TYPE* src,TYPE* out,const u64* xd,const u64* xs,u32 xr,u64 xo,const u64* id,const u64* is,u32 ir,u64 io,const u64* sd,const u64* ss,u32 sr,u64 so,u32 axis,u64 out_len,u64 idx_len){u64 gid=global_id();if(gid>=out_len)return;ACC acc=TO(x[view_offset(gid,xd,xs,xr,xo)]);for(u64 q=0;q<idx_len;q++){bool same=true;for(u32 a=0;a<xr;a++){u64 dc=logical_coord(gid,xd,xr,a),sc=logical_coord(q,id,ir,a);if(a==axis){i64 raw=idx[view_offset(q,id,is,ir,io)];if(raw<0||(u64)raw>=xd[axis]){same=false;break;}sc=(u64)raw;}if(dc!=sc)same=false;}if(same){u64 sb=so;for(u32 a=0;a<sr;a++)sb+=logical_coord(q,id,ir,a)*ss[a];acc=ADD(acc,TO(src[sb]));}}out[gid]=FROM(acc);}
INDEX_KERNELS(__half,float,f16,to_f32,from_f16,FLOAT_ADD)
INDEX_KERNELS(float,float,f32,to_f32,from_f32,FLOAT_ADD)
INDEX_KERNELS(i64,i64,i64,to_i64,from_i64,INT_ADD)

#define CONTIG_INDEX_KERNELS(TYPE,ACC,NAME,TO,FROM,ADD) \
extern "C" __global__ void index_select_axis0_##NAME(const TYPE* x,const i64* idx,TYPE* out,u64 len,u64 inner,u64 bound){u64 gid=global_id();if(gid<len){u64 q=gid/inner,c=gid%inner;i64 raw=idx[q];out[gid]=(raw<0||(u64)raw>=bound)?FROM((ACC)0):x[(u64)raw*inner+c];}} \
extern "C" __global__ void index_add_axis0_##NAME(const TYPE* x,const i64* idx,const TYPE* src,TYPE* out,u64 rows,u64 indices,u64 inner,u64 bound,const u64* sd,const u64* ss,u32 sr,u64 so){u64 row=global_id();if(row<rows){for(u64 c=0;c<inner;c++)out[row*inner+c]=x[row*inner+c];for(u64 q=0;q<indices;q++){i64 raw=idx[q];if(raw>=0&&(u64)raw<bound&&(u64)raw==row)for(u64 c=0;c<inner;c++){u64 pos=row*inner+c;out[pos]=FROM(ADD(TO(out[pos]),TO(src[view_offset(q*inner+c,sd,ss,sr,so)])));}}}} \
extern "C" __global__ void gather_last_##NAME(const TYPE* x,const i64* idx,TYPE* out,u64 len,u64 classes,u64 picks,u64 bound){u64 gid=global_id();if(gid<len){u64 row=gid/picks;i64 raw=idx[gid];out[gid]=(raw<0||(u64)raw>=bound)?FROM((ACC)0):x[row*classes+(u64)raw];}} \
extern "C" __global__ void scatter_add_last_##NAME(const TYPE* x,const i64* idx,const TYPE* src,TYPE* out,u64 len,u64 classes,u64 picks,u64 bound,const u64* sd,const u64* ss,u32 sr,u64 so){u64 gid=global_id();if(gid<len){u64 row=gid/classes,c=gid%classes;ACC acc=TO(x[gid]);for(u64 q=0;q<picks;q++){u64 pos=row*picks+q;i64 raw=idx[pos];if(raw>=0&&(u64)raw<bound&&(u64)raw==c)acc=ADD(acc,TO(src[view_offset(pos,sd,ss,sr,so)]));}out[gid]=FROM(acc);}}
CONTIG_INDEX_KERNELS(__half,float,f16,to_f32,from_f16,FLOAT_ADD)
CONTIG_INDEX_KERNELS(float,float,f32,to_f32,from_f32,FLOAT_ADD)
CONTIG_INDEX_KERNELS(i64,i64,i64,to_i64,from_i64,INT_ADD)

static __device__ __forceinline__ void atomic_accumulate(float* dst, float value) {
    atomicAdd(dst, value);
}
static __device__ __forceinline__ void atomic_accumulate(i64* dst, i64 value) {
    atomicAdd((unsigned long long*)dst, (unsigned long long)value);
}

#define ATOMIC_INDEX_KERNELS(TYPE,NAME) \
extern "C" __global__ void index_add_axis0_atomic_##NAME(const i64* idx,const TYPE* src,TYPE* out,u64 len,u64 inner,u64 bound,const u64* sd,const u64* ss,u32 sr,u64 so){u64 gid=global_id();if(gid<len){u64 q=gid/inner,c=gid%inner;i64 raw=idx[q];if(raw>=0&&(u64)raw<bound)atomic_accumulate(out+(u64)raw*inner+c,src[view_offset(gid,sd,ss,sr,so)]);}} \
extern "C" __global__ void scatter_add_last_atomic_##NAME(const i64* idx,const TYPE* src,TYPE* out,u64 len,u64 classes,u64 picks,u64 bound,const u64* sd,const u64* ss,u32 sr,u64 so){u64 gid=global_id();if(gid<len){u64 row=gid/picks;i64 raw=idx[gid];if(raw>=0&&(u64)raw<bound)atomic_accumulate(out+row*classes+(u64)raw,src[view_offset(gid,sd,ss,sr,so)]);}}
ATOMIC_INDEX_KERNELS(float,f32)
ATOMIC_INDEX_KERNELS(i64,i64)

extern "C" __global__ void index_select_bool(const u8* x,const i64* idx,u8* out,const u64* xd,const u64* xs,u32 xr,u64 xo,const u64* id,const u64* is,u32 ir,u64 io,const u64* od,u32 axis,u64 len){u64 gid=global_id();if(gid>=len)return;i64 raw=idx[view_offset(logical_coord(gid,od,xr,axis),id,is,ir,io)];if(raw<0||(u64)raw>=xd[axis]){out[gid]=0;return;}u64 base=xo;for(u32 a=0;a<xr;a++){u64 c=logical_coord(gid,od,xr,a);base+=(a==axis?(u64)raw:c)*xs[a];}out[gid]=x[base];}
extern "C" __global__ void gather_bool(const u8* x,const i64* idx,u8* out,const u64* xd,const u64* xs,u32 xr,u64 xo,const u64* id,const u64* is,u32 ir,u64 io,u32 axis,u64 len){u64 gid=global_id();if(gid>=len)return;i64 raw=idx[view_offset(gid,id,is,ir,io)];if(raw<0||(u64)raw>=xd[axis]){out[gid]=0;return;}u64 base=xo;for(u32 a=0;a<xr;a++){u64 c=logical_coord(gid,id,ir,a);base+=(a==axis?(u64)raw:c)*xs[a];}out[gid]=x[base];}

struct ConvParams {u64 n,ci,h,w,co,kh,kw,oh,ow,sh,sw,ph,pw,dh,dw;};
static __device__ __forceinline__ bool source_pos(u64 out,u64 window,u64 stride,u64 dilation,u64 pad,u64 size,u64* pos){u64 raw=out*stride+window*dilation;if(raw<pad)return false;*pos=raw-pad;return *pos<size;}
static __device__ __forceinline__ bool window_pos(u64 pos,u64 window,u64 stride,u64 dilation,u64 pad,u64 out_size,u64* out){u64 shifted=pos+pad,tap=window*dilation;if(shifted<tap)return false;u64 rel=shifted-tap;if(rel%stride!=0)return false;*out=rel/stride;return *out<out_size;}
static __device__ __forceinline__ bool window_range(u64 pos,u64 stride,u64 pad,u64 extent,u64 out_size,u64* lo,u64* hi){u64 shifted=pos+pad;*hi=shifted/stride;if(*hi>=out_size)*hi=out_size-1;*lo=0;if(shifted>=extent)*lo=(shifted-extent+stride)/stride;return *lo<=*hi;}

#define CONV_ARGS u64 pn,u64 pci,u64 ph,u64 pw,u64 pco,u64 pkh,u64 pkw,u64 poh,u64 pow,u64 psh,u64 psw,u64 pph,u64 ppw,u64 pdh,u64 pdw
#define CONV_INIT ConvParams p={pn,pci,ph,pw,pco,pkh,pkw,poh,pow,psh,psw,pph,ppw,pdh,pdw};
#define CONV_KERNELS(TYPE,ACC,NAME,TO,FROM,ADD,MUL,DIV) \
extern "C" __global__ void conv2d_##NAME(const TYPE* x,const TYPE* w,TYPE* out,const u64* xs,u64 xo,const u64* ws,u64 wo,CONV_ARGS,u64 len){CONV_INIT u64 gid=global_id();if(gid>=len)return;u64 ow=gid%p.ow,t=gid/p.ow,oh=t%p.oh;t/=p.oh;u64 oc=t%p.co,b=t/p.co;ACC acc=(ACC)0;for(u64 ic=0;ic<p.ci;ic++)for(u64 kh=0;kh<p.kh;kh++){u64 ih;if(!source_pos(oh,kh,p.sh,p.dh,p.ph,p.h,&ih))continue;for(u64 kw=0;kw<p.kw;kw++){u64 iw;if(source_pos(ow,kw,p.sw,p.dw,p.pw,p.w,&iw))acc=ADD(acc,MUL(TO(x[xo+b*xs[0]+ic*xs[1]+ih*xs[2]+iw*xs[3]]),TO(w[wo+oc*ws[0]+ic*ws[1]+kh*ws[2]+kw*ws[3]])));}}out[gid]=FROM(acc);} \
extern "C" __global__ void pool_##NAME(const TYPE* x,TYPE* out,const u64* xs,u64 xo,CONV_ARGS,u64 len,u32 op){CONV_INIT u64 gid=global_id();if(gid>=len)return;u64 ow=gid%p.ow,t=gid/p.ow,oh=t%p.oh;t/=p.oh;u64 c=t%p.ci,b=t/p.ci;ACC acc=(ACC)0;bool first=true;for(u64 kh=0;kh<p.kh;kh++){u64 ih;if(!source_pos(oh,kh,p.sh,1,p.ph,p.h,&ih))continue;for(u64 kw=0;kw<p.kw;kw++){u64 iw;if(!source_pos(ow,kw,p.sw,1,p.pw,p.w,&iw))continue;ACC v=TO(x[xo+b*xs[0]+c*xs[1]+ih*xs[2]+iw*xs[3]]);if(op==0){if(first||(!value_nan(acc)&&(value_nan(v)||v>acc)))acc=v;}else acc=ADD(acc,v);first=false;}}if(op==1)acc=DIV(acc,(ACC)(p.kh*p.kw));out[gid]=FROM(acc);} \
extern "C" __global__ void conv_input_grad_##NAME(const TYPE* g,const TYPE* w,TYPE* out,const u64* gs,u64 go,const u64* ws,u64 wo,CONV_ARGS,u64 len){CONV_INIT u64 gid=global_id();if(gid>=len)return;u64 iw=gid%p.w,t=gid/p.w,ih=t%p.h;t/=p.h;u64 ic=t%p.ci,b=t/p.ci;ACC acc=(ACC)0;for(u64 kh=0;kh<p.kh;kh++){u64 oh;if(!window_pos(ih,kh,p.sh,p.dh,p.ph,p.oh,&oh))continue;for(u64 kw=0;kw<p.kw;kw++){u64 ow;if(!window_pos(iw,kw,p.sw,p.dw,p.pw,p.ow,&ow))continue;for(u64 oc=0;oc<p.co;oc++)acc=ADD(acc,MUL(TO(g[go+b*gs[0]+oc*gs[1]+oh*gs[2]+ow*gs[3]]),TO(w[wo+oc*ws[0]+ic*ws[1]+kh*ws[2]+kw*ws[3]])));}}out[gid]=FROM(acc);} \
extern "C" __global__ void conv_weight_grad_##NAME(const TYPE* g,const TYPE* x,TYPE* out,const u64* gs,u64 go,const u64* xs,u64 xo,CONV_ARGS,u64 len){CONV_INIT u64 gid=global_id();if(gid>=len)return;u64 kw=gid%p.kw,t=gid/p.kw,kh=t%p.kh;t/=p.kh;u64 ic=t%p.ci,oc=t/p.ci;ACC acc=(ACC)0;for(u64 b=0;b<p.n;b++)for(u64 oh=0;oh<p.oh;oh++){u64 ih;if(!source_pos(oh,kh,p.sh,p.dh,p.ph,p.h,&ih))continue;for(u64 ow=0;ow<p.ow;ow++){u64 iw;if(source_pos(ow,kw,p.sw,p.dw,p.pw,p.w,&iw))acc=ADD(acc,MUL(TO(g[go+b*gs[0]+oc*gs[1]+oh*gs[2]+ow*gs[3]]),TO(x[xo+b*xs[0]+ic*xs[1]+ih*xs[2]+iw*xs[3]])));}}out[gid]=FROM(acc);} \
extern "C" __global__ void pool_backward_##NAME(const TYPE* g,const TYPE* x,TYPE* out,const u64* gs,u64 go,const u64* xs,u64 xo,CONV_ARGS,u64 len,u32 op){CONV_INIT u64 gid=global_id();if(gid>=len)return;u64 iw=gid%p.w,t=gid/p.w,ih=t%p.h;t/=p.h;u64 c=t%p.ci,b=t/p.ci;ACC acc=(ACC)0,peak;u64 ohlo,ohhi,owlo,owhi;if(!window_range(ih,p.sh,p.ph,p.kh,p.oh,&ohlo,&ohhi)||!window_range(iw,p.sw,p.pw,p.kw,p.ow,&owlo,&owhi)){out[gid]=FROM(acc);return;}for(u64 oh=ohlo;oh<=ohhi;oh++)for(u64 ow=owlo;ow<=owhi;ow++){bool owns=op==1,first=true;u64 besth=0,bestw=0;peak=(ACC)0;for(u64 kh=0;kh<p.kh;kh++){u64 sh;if(!source_pos(oh,kh,p.sh,1,p.ph,p.h,&sh))continue;for(u64 kw=0;kw<p.kw;kw++){u64 sw;if(!source_pos(ow,kw,p.sw,1,p.pw,p.w,&sw))continue;ACC v=TO(x[xo+b*xs[0]+c*xs[1]+sh*xs[2]+sw*xs[3]]);if(first||(!value_nan(peak)&&(value_nan(v)||v>peak))){peak=v;besth=sh;bestw=sw;}first=false;}}if(op==0)owns=besth==ih&&bestw==iw;else owns=owns&&ih+p.ph>=oh*p.sh&&ih+p.ph<oh*p.sh+p.kh&&iw+p.pw>=ow*p.sw&&iw+p.pw<ow*p.sw+p.kw;if(owns){ACC v=TO(g[go+b*gs[0]+c*gs[1]+oh*gs[2]+ow*gs[3]]);acc=ADD(acc,op==1?DIV(v,(ACC)(p.kh*p.kw)):v);}}out[gid]=FROM(acc);}
CONV_KERNELS(__half,float,f16,to_f32,from_f16,FLOAT_ADD,FLOAT_MUL,FLOAT_DIV)
CONV_KERNELS(float,float,f32,to_f32,from_f32,FLOAT_ADD,FLOAT_MUL,FLOAT_DIV)
CONV_KERNELS(i64,i64,i64,to_i64,from_i64,INT_ADD,INT_MUL,INT_DIV)

#define FUSED_FLOAT(TYPE,NAME,TO,FROM) \
extern "C" __global__ void softmax_##NAME(const TYPE* x,TYPE* out,const u64* d,const u64* s,u32 r,u64 off,u64 rows,u64 width){u64 row=global_id();if(row>=rows)return;u64 base=reduced_offset(row,d,s,r,r-1,off);float peak=-INFINITY;bool nan=false;for(u64 c=0;c<width;c++){float v=TO(x[base+c*s[r-1]]);nan|=isnan(v);peak=fmaxf(peak,v);}if(nan){for(u64 c=0;c<width;c++)out[row*width+c]=FROM(NAN);return;}if(peak==-INFINITY){for(u64 c=0;c<width;c++)out[row*width+c]=FROM(0);return;}float sum=0;for(u64 c=0;c<width;c++)sum+=expf(TO(x[base+c*s[r-1]])-peak);for(u64 c=0;c<width;c++)out[row*width+c]=FROM(expf(TO(x[base+c*s[r-1]])-peak)/sum);} \
extern "C" __global__ void layer_norm_##NAME(const TYPE* x,const TYPE* w,const TYPE* b,TYPE* out,float* xhat,float* invout,const u64* xd,const u64* xs,u32 xr,u64 xo,const u64* ws,u64 woff,const u64* bs,u64 boff,u64 rows,u64 width,float eps,u32 save){u64 row=global_id();if(row>=rows)return;u64 base=reduced_offset(row,xd,xs,xr,xr-1,xo);float mean=0;for(u64 c=0;c<width;c++)mean+=TO(x[base+c*xs[xr-1]]);mean/=(float)width;float var=0;for(u64 c=0;c<width;c++){float z=TO(x[base+c*xs[xr-1]])-mean;var+=z*z;}float inv=rsqrtf(var/(float)width+eps);if(save)invout[row]=inv;for(u64 c=0;c<width;c++){float z=(TO(x[base+c*xs[xr-1]])-mean)*inv;if(save)xhat[row*width+c]=z;out[row*width+c]=FROM(z*TO(w[woff+c*ws[0]])+TO(b[boff+c*bs[0]]));}} \
extern "C" __global__ void layer_norm_backward_##NAME(const TYPE* g,const float* xhat,const float* inv,const TYPE* w,TYPE* out,const u64* gd,const u64* gs,u32 gr,u64 go,const u64* hd,const u64* hs,u32 hr,u64 ho,const u64* id,const u64* is,u32 ir,u64 io,const u64* ws,u64 wo,u64 rows,u64 width){u64 row=global_id();if(row>=rows)return;float sum=0,sumh=0;for(u64 c=0;c<width;c++){float dy=TO(g[view_offset(row*width+c,gd,gs,gr,go)])*TO(w[wo+c*ws[0]]),h=xhat[view_offset(row*width+c,hd,hs,hr,ho)];sum+=dy;sumh+=dy*h;}float iv=inv[view_offset(row,id,is,ir,io)];for(u64 c=0;c<width;c++){float dy=TO(g[view_offset(row*width+c,gd,gs,gr,go)])*TO(w[wo+c*ws[0]]),h=xhat[view_offset(row*width+c,hd,hs,hr,ho)];out[row*width+c]=FROM(iv*(dy-(sum+h*sumh)/(float)width));}}
FUSED_FLOAT(__half,f16,to_f32,from_f16)
FUSED_FLOAT(float,f32,to_f32,from_f32)

static __device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp[8];
    for (u32 delta=16;delta>0;delta>>=1) value+=__shfl_down_sync(0xffffffffu,value,delta);
    if ((threadIdx.x&31)==0) warp[threadIdx.x>>5]=value;
    __syncthreads();
    value=threadIdx.x<8?warp[threadIdx.x]:0.0f;
    if (threadIdx.x<32) for (u32 delta=16;delta>0;delta>>=1) value+=__shfl_down_sync(0xffffffffu,value,delta);
    if (threadIdx.x==0) warp[0]=value;
    __syncthreads();
    return warp[0];
}
static __device__ __forceinline__ float block_max(float value) {
    __shared__ float warp[8];
    for (u32 delta=16;delta>0;delta>>=1) value=fmaxf(value,__shfl_down_sync(0xffffffffu,value,delta));
    if ((threadIdx.x&31)==0) warp[threadIdx.x>>5]=value;
    __syncthreads();
    value=threadIdx.x<8?warp[threadIdx.x]:-INFINITY;
    if (threadIdx.x<32) for (u32 delta=16;delta>0;delta>>=1) value=fmaxf(value,__shfl_down_sync(0xffffffffu,value,delta));
    if (threadIdx.x==0) warp[0]=value;
    __syncthreads();
    return warp[0];
}
static __device__ __forceinline__ float block_min(float value) {
    __shared__ float warp[8];
    for (u32 delta=16;delta>0;delta>>=1) value=fminf(value,__shfl_down_sync(0xffffffffu,value,delta));
    if ((threadIdx.x&31)==0) warp[threadIdx.x>>5]=value;
    __syncthreads();
    value=threadIdx.x<8?warp[threadIdx.x]:INFINITY;
    if (threadIdx.x<32) for (u32 delta=16;delta>0;delta>>=1) value=fminf(value,__shfl_down_sync(0xffffffffu,value,delta));
    if (threadIdx.x==0) warp[0]=value;
    __syncthreads();
    return warp[0];
}

#define PARALLEL_FLOAT(TYPE,NAME,TO,FROM) \
extern "C" __global__ void reduce_parallel_##NAME(const TYPE* x,TYPE* out,const u64* d,const u64* s,u32 r,u64 off,u64 out_len,u32 axis,u32 op){u64 row=blockIdx.x;if(row>=out_len)return;u64 n=d[axis],base=reduced_offset(row,d,s,r,axis,off);float acc=op<2?0.0f:(op==2?-INFINITY:INFINITY),nan=0.0f;for(u64 i=threadIdx.x;i<n;i+=blockDim.x){float v=TO(x[base+i*s[axis]]);nan+=isnan(v)?1.0f:0.0f;if(op<2)acc+=v;else if(op==2)acc=fmaxf(acc,v);else acc=fminf(acc,v);}nan=block_sum(nan);acc=op<2?block_sum(acc):(op==2?block_max(acc):block_min(acc));if(threadIdx.x==0){float v=nan>0.0f?NAN:acc;if(op==1)v/=(float)n;out[row]=FROM(v);}} \
extern "C" __global__ void softmax_parallel_##NAME(const TYPE* x,TYPE* out,const u64* d,const u64* s,u32 r,u64 off,u64 rows,u64 width){u64 row=blockIdx.x;if(row>=rows)return;u64 base=reduced_offset(row,d,s,r,r-1,off);float peak=-INFINITY,nan=0.0f;for(u64 c=threadIdx.x;c<width;c+=blockDim.x){float v=TO(x[base+c*s[r-1]]);nan+=isnan(v)?1.0f:0.0f;peak=fmaxf(peak,v);}peak=block_max(peak);nan=block_sum(nan);float sum=0.0f;if(nan==0.0f&&peak!=-INFINITY)for(u64 c=threadIdx.x;c<width;c+=blockDim.x)sum+=expf(TO(x[base+c*s[r-1]])-peak);sum=block_sum(sum);for(u64 c=threadIdx.x;c<width;c+=blockDim.x)out[row*width+c]=nan>0.0f?FROM(NAN):(peak==-INFINITY?FROM(0.0f):FROM(expf(TO(x[base+c*s[r-1]])-peak)/sum));} \
extern "C" __global__ void layer_norm_parallel_##NAME(const TYPE* x,const TYPE* w,const TYPE* b,TYPE* out,float* xhat,float* invout,const u64* xd,const u64* xs,u32 xr,u64 xo,const u64* ws,u64 woff,const u64* bs,u64 boff,u64 rows,u64 width,float eps,u32 save){u64 row=blockIdx.x;if(row>=rows)return;u64 base=reduced_offset(row,xd,xs,xr,xr-1,xo);float sum=0.0f;for(u64 c=threadIdx.x;c<width;c+=blockDim.x)sum+=TO(x[base+c*xs[xr-1]]);float mean=block_sum(sum)/(float)width,var=0.0f;for(u64 c=threadIdx.x;c<width;c+=blockDim.x){float z=TO(x[base+c*xs[xr-1]])-mean;var+=z*z;}float inv=rsqrtf(block_sum(var)/(float)width+eps);if(save&&threadIdx.x==0)invout[row]=inv;for(u64 c=threadIdx.x;c<width;c+=blockDim.x){float z=(TO(x[base+c*xs[xr-1]])-mean)*inv;if(save)xhat[row*width+c]=z;out[row*width+c]=FROM(z*TO(w[woff+c*ws[0]])+TO(b[boff+c*bs[0]]));}} \
extern "C" __global__ void layer_norm_backward_parallel_##NAME(const TYPE* g,const float* xhat,const float* inv,const TYPE* w,TYPE* out,const u64* gd,const u64* gs,u32 gr,u64 go,const u64* hd,const u64* hs,u32 hr,u64 ho,const u64* id,const u64* is,u32 ir,u64 io,const u64* ws,u64 wo,u64 rows,u64 width){u64 row=blockIdx.x;if(row>=rows)return;float sum=0.0f,sumh=0.0f;for(u64 c=threadIdx.x;c<width;c+=blockDim.x){float dy=TO(g[view_offset(row*width+c,gd,gs,gr,go)])*TO(w[wo+c*ws[0]]),h=xhat[view_offset(row*width+c,hd,hs,hr,ho)];sum+=dy;sumh+=dy*h;}sum=block_sum(sum);sumh=block_sum(sumh);float iv=inv[view_offset(row,id,is,ir,io)];for(u64 c=threadIdx.x;c<width;c+=blockDim.x){float dy=TO(g[view_offset(row*width+c,gd,gs,gr,go)])*TO(w[wo+c*ws[0]]),h=xhat[view_offset(row*width+c,hd,hs,hr,ho)];out[row*width+c]=FROM(iv*(dy-(sum+h*sumh)/(float)width));}}
PARALLEL_FLOAT(__half,f16,to_f32,from_f16)
PARALLEL_FLOAT(float,f32,to_f32,from_f32)

#define OPT_KERNELS(TYPE,NAME,TO,FROM) \
extern "C" __global__ void sgd_##NAME(const TYPE* p,const TYPE* g,const float* vin,TYPE* pout,float* vout,u64 len,const float* hp,u32 hasv,u32 usem){u64 gid=global_id();if(gid>=len)return;float pv=TO(p[gid]),grad=TO(g[gid])+hp[2]*pv,dir=hasv?hp[1]*vin[gid]+grad:grad;if(usem)vout[gid]=dir;pout[gid]=FROM(pv-hp[0]*dir);} \
extern "C" __global__ void adam_##NAME(const TYPE* p,const TYPE* g,const float* mi,const float* vi,TYPE* po,float* mo,float* vo,u64 len,const float* h){u64 gid=global_id();if(gid>=len)return;float pv=TO(p[gid]),grad=TO(g[gid]);bool dec=h[7]==1.0f;if(!dec)grad+=h[4]*pv;float m=h[1]*mi[gid]+(1-h[1])*grad,v=h[2]*vi[gid]+(1-h[2])*grad*grad,next=dec?pv*(1-h[0]*h[4]):pv;next-=h[0]*(m/h[5])/(sqrtf(v/h[6])+h[3]);po[gid]=FROM(next);mo[gid]=m;vo[gid]=v;}
OPT_KERNELS(__half,f16,to_f32,from_f16)
OPT_KERNELS(float,f32,to_f32,from_f32)
