#include <metal_stdlib>
using namespace metal;

static ulong view_offset(
    ulong logical,
    constant ulong* dims,
    constant ulong* strides,
    uint rank,
    ulong offset
) {
    for (uint axis = rank; axis > 0; --axis) {
        ulong dim = dims[axis - 1];
        if (dim != 0) {
            offset += (logical % dim) * strides[axis - 1];
            logical /= dim;
        }
    }
    return offset;
}

#define COPY_KERNEL(TYPE, NAME) \
kernel void copy_##NAME( \
    device const TYPE* src [[buffer(0)]], \
    device TYPE* dst [[buffer(1)]], \
    constant ulong* dims [[buffer(2)]], \
    constant ulong* strides [[buffer(3)]], \
    constant uint& rank [[buffer(4)]], \
    constant ulong& offset [[buffer(5)]], \
    constant ulong& len [[buffer(6)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid < len) dst[gid] = src[view_offset(gid, dims, strides, rank, offset)]; \
} \
kernel void copy_into_##NAME( \
    device const TYPE* src [[buffer(0)]], \
    device TYPE* dst [[buffer(1)]], \
    constant ulong* src_dims [[buffer(2)]], \
    constant ulong* src_strides [[buffer(3)]], \
    constant uint& src_rank [[buffer(4)]], \
    constant ulong& src_offset [[buffer(5)]], \
    constant ulong* dst_dims [[buffer(6)]], \
    constant ulong* dst_strides [[buffer(7)]], \
    constant uint& dst_rank [[buffer(8)]], \
    constant ulong& dst_offset [[buffer(9)]], \
    constant ulong& len [[buffer(10)]], \
    uint gid [[thread_position_in_grid]]) { \
    if (gid < len) { \
        ulong from = view_offset(gid, src_dims, src_strides, src_rank, src_offset); \
        ulong to = view_offset(gid, dst_dims, dst_strides, dst_rank, dst_offset); \
        dst[to] = src[from]; \
    } \
}

COPY_KERNEL(half, f16)
COPY_KERNEL(float, f32)
COPY_KERNEL(long, i64)
COPY_KERNEL(uchar, bool)

static ulong logical_coord(ulong logical, constant ulong* dims, uint rank, uint wanted) {
    for (uint axis = rank; axis > 0; --axis) {
        ulong dim = dims[axis - 1];
        ulong coord = dim == 0 ? 0 : logical % dim;
        if (axis - 1 == wanted) return coord;
        if (dim != 0) logical /= dim;
    }
    return 0;
}

static ulong reduced_offset(
    ulong outer, constant ulong* dims, constant ulong* strides,
    uint rank, uint reduced_axis, ulong offset
) {
    for (uint axis = rank; axis > 0; --axis) {
        uint a = axis - 1;
        if (a == reduced_axis) continue;
        ulong dim = dims[a];
        if (dim != 0) {
            offset += (outer % dim) * strides[a];
            outer /= dim;
        }
    }
    return offset;
}

static bool value_nan(half value) { return isnan(value); }
static bool value_nan(float value) { return isnan(value); }
static bool value_nan(long value) { (void)value; return false; }

kernel void validate_indices(
    device const long* idx [[buffer(0)]], device long* result [[buffer(1)]],
    constant ulong* dims [[buffer(2)]], constant ulong* strides [[buffer(3)]],
    constant uint& rank [[buffer(4)]], constant ulong& offset [[buffer(5)]],
    constant ulong& len [[buffer(6)]], constant ulong& bound [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    result[0] = 0;
    result[1] = 0;
    for (ulong i = 0; i < len; ++i) {
        long value = idx[view_offset(i, dims, strides, rank, offset)];
        if (value < 0 || ulong(value) >= bound) {
            result[0] = 1;
            result[1] = value;
            return;
        }
    }
}

#define NUMERIC_KERNELS(TYPE, ACC, NAME, TO_ACC, FROM_ACC, ADD, SUB, MUL, DIV) \
kernel void binary_##NAME( \
    device const TYPE* lhs [[buffer(0)]], device const TYPE* rhs [[buffer(1)]], \
    device TYPE* out [[buffer(2)]], constant ulong* ld [[buffer(3)]], \
    constant ulong* ls [[buffer(4)]], constant uint& lr [[buffer(5)]], \
    constant ulong& lo [[buffer(6)]], constant ulong* rd [[buffer(7)]], \
    constant ulong* rs [[buffer(8)]], constant uint& rr [[buffer(9)]], \
    constant ulong& ro [[buffer(10)]], constant ulong& len [[buffer(11)]], \
    constant uint& op [[buffer(12)]], uint gid [[thread_position_in_grid]]) { \
    if (gid >= len) return; \
    TYPE a = lhs[view_offset(gid, ld, ls, lr, lo)]; \
    TYPE b = rhs[view_offset(gid, rd, rs, rr, ro)]; \
    switch (op) { case 0: out[gid]=ADD(a,b); break; case 1: out[gid]=SUB(a,b); break; \
      case 2: out[gid]=MUL(a,b); break; case 3: out[gid]=DIV(a,b); break; \
      case 4: out[gid]=value_nan(a)?b:(value_nan(b)?a:(a>=b?a:b)); break; \
      default: out[gid]=value_nan(a)?b:(value_nan(b)?a:(a<=b?a:b)); } \
} \
kernel void scalar_##NAME( \
    device const TYPE* x [[buffer(0)]], device TYPE* out [[buffer(1)]], \
    constant ulong* d [[buffer(2)]], constant ulong* s [[buffer(3)]], \
    constant uint& r [[buffer(4)]], constant ulong& off [[buffer(5)]], \
    constant ulong& len [[buffer(6)]], constant ACC& scalar [[buffer(7)]], \
    constant uint& op [[buffer(8)]], uint gid [[thread_position_in_grid]]) { \
    if (gid >= len) return; ACC a=TO_ACC(x[view_offset(gid,d,s,r,off)]), b=scalar, v; \
    switch(op){case 0:v=ADD(a,b);break;case 1:v=SUB(a,b);break;case 2:v=MUL(a,b);break; \
      case 3:v=DIV(a,b);break;case 4:v=value_nan(a)?b:(value_nan(b)?a:(a>=b?a:b));break;default:v=value_nan(a)?b:(value_nan(b)?a:(a<=b?a:b));} \
    out[gid]=FROM_ACC(v); \
} \
kernel void compare_##NAME( \
    device const TYPE* lhs [[buffer(0)]], device const TYPE* rhs [[buffer(1)]], \
    device uchar* out [[buffer(2)]], constant ulong* ld [[buffer(3)]], \
    constant ulong* ls [[buffer(4)]], constant uint& lr [[buffer(5)]], \
    constant ulong& lo [[buffer(6)]], constant ulong* rd [[buffer(7)]], \
    constant ulong* rs [[buffer(8)]], constant uint& rr [[buffer(9)]], \
    constant ulong& ro [[buffer(10)]], constant ulong& len [[buffer(11)]], \
    constant uint& op [[buffer(12)]], uint gid [[thread_position_in_grid]]) { \
    if(gid>=len)return; TYPE a=lhs[view_offset(gid,ld,ls,lr,lo)],b=rhs[view_offset(gid,rd,rs,rr,ro)]; \
    bool v; switch(op){case 0:v=a==b;break;case 1:v=a!=b;break;case 2:v=a<b;break; \
      case 3:v=a<=b;break;case 4:v=a>b;break;default:v=a>=b;} out[gid]=uchar(v); \
} \
kernel void reduce_##NAME( \
    device const TYPE* x [[buffer(0)]], device TYPE* out [[buffer(1)]], \
    constant ulong* d [[buffer(2)]], constant ulong* s [[buffer(3)]], \
    constant uint& r [[buffer(4)]], constant ulong& off [[buffer(5)]], \
    constant ulong& out_len [[buffer(6)]], constant uint& axis [[buffer(7)]], \
    constant uint& op [[buffer(8)]], uint gid [[thread_position_in_grid]]) { \
    if(gid>=out_len)return; ulong base=reduced_offset(gid,d,s,r,axis,off), n=d[axis]; \
    ACC acc=ACC(0); for(ulong i=0;i<n;i++){ACC v=TO_ACC(x[base+i*s[axis]]); \
      if(op<2)acc+=v;else if(i==0)acc=v;else if(value_nan(acc)||value_nan(v))acc=acc+v;else if(op==2)acc=acc>=v?acc:v;else acc=acc<=v?acc:v;} \
    if(op==1)acc/=ACC(n); out[gid]=FROM_ACC(acc); \
} \
kernel void arg_reduce_##NAME( \
    device const TYPE* x [[buffer(0)]], device long* out [[buffer(1)]], \
    constant ulong* d [[buffer(2)]], constant ulong* s [[buffer(3)]], \
    constant uint& r [[buffer(4)]], constant ulong& off [[buffer(5)]], \
    constant ulong& out_len [[buffer(6)]], constant uint& axis [[buffer(7)]], \
    constant uint& op [[buffer(8)]], uint gid [[thread_position_in_grid]]) { \
    if(gid>=out_len)return; ulong base=reduced_offset(gid,d,s,r,axis,off),n=d[axis],best=0; \
    TYPE bv=x[base]; for(ulong i=1;i<n;i++){TYPE v=x[base+i*s[axis]]; \
      bool better=op==0?(value_nan(bv)&&!value_nan(v)):(value_nan(v)&&!value_nan(bv)); \
      if(!value_nan(v)&&!value_nan(bv))better=op==0?v>bv:v<bv;if(better){bv=v;best=i;}} out[gid]=long(best); \
} \
kernel void matmul_##NAME( \
    device const TYPE* lhs [[buffer(0)]], device const TYPE* rhs [[buffer(1)]], device TYPE* out [[buffer(2)]], \
    constant ulong* bd [[buffer(3)]], constant ulong* lbs [[buffer(4)]], constant ulong* rbs [[buffer(5)]], \
    constant uint& br [[buffer(6)]], constant ulong* p [[buffer(7)]], constant ulong& len [[buffer(8)]], \
    uint gid [[thread_position_in_grid]]) { \
    if(gid>=len)return; ulong m=p[0],k=p[1],n=p[2],lo=p[3],ro=p[4]; ulong batch=gid/(m*n),within=gid%(m*n); \
    ulong li=lo,ri=ro,b=batch; for(uint a=br;a>0;--a){ulong dim=bd[a-1],c=dim==0?0:b%dim;if(dim)b/=dim;li+=c*lbs[a-1];ri+=c*rbs[a-1];} \
    ulong row=within/n,col=within%n; ACC acc=ACC(0); for(ulong q=0;q<k;q++)acc+=TO_ACC(lhs[li+row*p[5]+q*p[6]])*TO_ACC(rhs[ri+q*p[7]+col*p[8]]); \
    out[gid]=FROM_ACC(acc); \
}

#define IDENTITY_F32(x) float(x)
#define IDENTITY_I64(x) long(x)
#define FROM_F16(x) half(x)
#define FROM_F32(x) float(x)
#define FROM_I64(x) long(x)
#define NORMAL_ADD(a,b) ((a)+(b))
#define NORMAL_SUB(a,b) ((a)-(b))
#define NORMAL_MUL(a,b) ((a)*(b))
#define FLOAT_DIV(a,b) ((a)/(b))
#define LONG_LOW as_type<long>(0x8000000000000000UL)
#define WRAP_ADD(a,b) as_type<long>(as_type<ulong>(a)+as_type<ulong>(b))
#define WRAP_SUB(a,b) as_type<long>(as_type<ulong>(a)-as_type<ulong>(b))
#define WRAP_MUL(a,b) as_type<long>(as_type<ulong>(a)*as_type<ulong>(b))
#define INT_DIV(a,b) ((b)==0?0:((a)==LONG_LOW&&(b)==-1?LONG_LOW:(a)/(b)))
NUMERIC_KERNELS(half, float, f16, IDENTITY_F32, FROM_F16, NORMAL_ADD, NORMAL_SUB, NORMAL_MUL, FLOAT_DIV)
NUMERIC_KERNELS(float, float, f32, IDENTITY_F32, FROM_F32, NORMAL_ADD, NORMAL_SUB, NORMAL_MUL, FLOAT_DIV)
NUMERIC_KERNELS(long, long, i64, IDENTITY_I64, FROM_I64, WRAP_ADD, WRAP_SUB, WRAP_MUL, INT_DIV)

kernel void compare_bool(
    device const uchar* lhs [[buffer(0)]], device const uchar* rhs [[buffer(1)]],
    device uchar* out [[buffer(2)]], constant ulong* ld [[buffer(3)]],
    constant ulong* ls [[buffer(4)]], constant uint& lr [[buffer(5)]],
    constant ulong& lo [[buffer(6)]], constant ulong* rd [[buffer(7)]],
    constant ulong* rs [[buffer(8)]], constant uint& rr [[buffer(9)]],
    constant ulong& ro [[buffer(10)]], constant ulong& len [[buffer(11)]],
    constant uint& op [[buffer(12)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= len) return;
    uchar a=lhs[view_offset(gid,ld,ls,lr,lo)], b=rhs[view_offset(gid,rd,rs,rr,ro)];
    bool v; switch(op){case 0:v=a==b;break;case 1:v=a!=b;break;case 2:v=a<b;break;
      case 3:v=a<=b;break;case 4:v=a>b;break;default:v=a>=b;} out[gid]=uchar(v);
}

// Single-precision fdlibm rational forms on the cancellation-sensitive core;
// the asymptotic branch is only used once GELU is already in its rounded tail.
static float erf_accurate(float x) {
    float ax = fabs(x), value;
    if (ax < 0.84375f) {
        float z = ax * ax;
        float r = 0.1283791671f + z * (-0.3250421073f + z * (-0.0284817498f
            + z * (-0.0057702702f + z * -0.0000237630f)));
        float s = 1.0f + z * (0.3979172111f + z * (0.0650222525f
            + z * (0.0050813062f + z * (0.0001324947f + z * -0.0000039602f))));
        value = ax + ax * (r / s);
    } else if (ax < 1.25f) {
        float z = ax - 1.0f;
        float p = -0.0023621186f + z * (0.4148561060f + z * (-0.3722078800f
            + z * (0.3183466196f + z * (-0.1108946949f
            + z * (0.0354783051f + z * -0.0021663755f)))));
        float q = 1.0f + z * (0.1064208820f + z * (0.5403979421f
            + z * (0.0718286559f + z * (0.1261712164f
            + z * (0.0136370836f + z * 0.0119845001f)))));
        value = 0.8450629115f + p / q;
    } else {
        float t = 1.0f / (1.0f + 0.5f * ax);
        float tau = t * exp(-ax * ax - 1.26551223f + t * (1.00002368f
            + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f
            + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f
            + t * (-0.82215223f + t * 0.17087277f)))))))));
        value = 1.0f - tau;
    }
    return x < 0.0f ? -value : value;
}

#define FLOAT_UNARY(TYPE, NAME, FROM) \
kernel void unary_##NAME(device const TYPE* x [[buffer(0)]],device TYPE* out [[buffer(1)]], \
 constant ulong* d [[buffer(2)]],constant ulong* s [[buffer(3)]],constant uint& r [[buffer(4)]], \
 constant ulong& off [[buffer(5)]],constant ulong& len [[buffer(6)]],constant uint& op [[buffer(7)]],uint gid [[thread_position_in_grid]]){ \
 if(gid>=len)return;float v=float(x[view_offset(gid,d,s,r,off)]),z;switch(op){case 0:z=max(v,0.0f);break; \
 case 1:z=0.5f*v*(1.0f+erf_accurate(v*0.7071067811865475f));break;case 2:z=exp(v);break;case 3:z=log(v);break; \
 case 4:z=sqrt(v);break;case 5:z=tanh(v);break;case 6:z=1.0f/(1.0f+exp(-v));break;case 7:z=-v;break;default:z=fabs(v);}out[gid]=FROM(z);}
FLOAT_UNARY(half,f16,FROM_F16)
FLOAT_UNARY(float,f32,FROM_F32)

kernel void unary_i64(device const long* x [[buffer(0)]],device long* out [[buffer(1)]],
 constant ulong* d [[buffer(2)]],constant ulong* s [[buffer(3)]],constant uint& r [[buffer(4)]],
 constant ulong& off [[buffer(5)]],constant ulong& len [[buffer(6)]],constant uint& op [[buffer(7)]],uint gid [[thread_position_in_grid]]){
 if(gid>=len)return;long v=x[view_offset(gid,d,s,r,off)];long neg=as_type<long>(0UL-as_type<ulong>(v));out[gid]=op==7?neg:(v<0?neg:v);
}

#define SELECT_KERNELS(TYPE, NAME, FROM, FILL) \
kernel void where_##NAME(device const uchar* c [[buffer(0)]],device const TYPE* t [[buffer(1)]],device const TYPE* f [[buffer(2)]],device TYPE* out [[buffer(3)]], \
 constant ulong* cd [[buffer(4)]],constant ulong* cs [[buffer(5)]],constant uint& cr [[buffer(6)]],constant ulong& co [[buffer(7)]], \
 constant ulong* td [[buffer(8)]],constant ulong* ts [[buffer(9)]],constant uint& tr [[buffer(10)]],constant ulong& to [[buffer(11)]], \
 constant ulong* fd [[buffer(12)]],constant ulong* fs [[buffer(13)]],constant uint& fr [[buffer(14)]],constant ulong& fo [[buffer(15)]], \
 constant ulong& len [[buffer(16)]],uint gid [[thread_position_in_grid]]){if(gid<len)out[gid]=c[view_offset(gid,cd,cs,cr,co)]?t[view_offset(gid,td,ts,tr,to)]:f[view_offset(gid,fd,fs,fr,fo)];} \
kernel void masked_##NAME(device const TYPE* x [[buffer(0)]],device const uchar* m [[buffer(1)]],device TYPE* out [[buffer(2)]], \
 constant ulong* xd [[buffer(3)]],constant ulong* xs [[buffer(4)]],constant uint& xr [[buffer(5)]],constant ulong& xo [[buffer(6)]], \
 constant ulong* md [[buffer(7)]],constant ulong* ms [[buffer(8)]],constant uint& mr [[buffer(9)]],constant ulong& mo [[buffer(10)]], \
 constant ulong& len [[buffer(11)]],constant FILL& fill [[buffer(12)]],uint gid [[thread_position_in_grid]]){if(gid<len)out[gid]=m[view_offset(gid,md,ms,mr,mo)]?FROM(fill):x[view_offset(gid,xd,xs,xr,xo)];}
SELECT_KERNELS(half,f16,FROM_F16,float)
SELECT_KERNELS(float,f32,FROM_F32,float)
SELECT_KERNELS(long,i64,FROM_I64,long)
SELECT_KERNELS(uchar,bool,uchar,float)

#define CAST_KERNEL(FROM, TO, FN, TN, EXPR) \
kernel void cast_##FN##_to_##TN(device const FROM* x [[buffer(0)]],device TO* out [[buffer(1)]], \
 constant ulong* d [[buffer(2)]],constant ulong* s [[buffer(3)]],constant uint& r [[buffer(4)]],constant ulong& off [[buffer(5)]], \
 constant ulong& len [[buffer(6)]],uint gid [[thread_position_in_grid]]){if(gid<len){FROM v=x[view_offset(gid,d,s,r,off)];out[gid]=(EXPR);}}
static long float_to_i64(float value) {
    if (isnan(value)) return 0;
    // f32 cannot represent i64::MAX. 2^63 is the first source value at or
    // above it, while -2^63 is exactly representable.
    if (value >= 0x1p63f) return as_type<long>(0x7fffffffffffffffUL);
    if (value <= -0x1p63f) return as_type<long>(0x8000000000000000UL);
    return long(value);
}
CAST_KERNEL(half,float,f16,f32,float(v))
CAST_KERNEL(float,half,f32,f16,half(v))
CAST_KERNEL(half,long,f16,i64,float_to_i64(float(v)))
CAST_KERNEL(long,half,i64,f16,half(v))
CAST_KERNEL(float,long,f32,i64,float_to_i64(v))
CAST_KERNEL(long,float,i64,f32,float(v))
CAST_KERNEL(half,uchar,f16,bool,uchar(v!=half(0)))
CAST_KERNEL(uchar,half,bool,f16,half(v!=0))
CAST_KERNEL(float,uchar,f32,bool,uchar(v!=0.0f))
CAST_KERNEL(uchar,float,bool,f32,float(v!=0))
CAST_KERNEL(long,uchar,i64,bool,uchar(v!=0))
CAST_KERNEL(uchar,long,bool,i64,long(v!=0))

#define INDEX_KERNELS(TYPE, ACC, NAME, TO_ACC, FROM_ACC) \
kernel void index_select_##NAME(device const TYPE* x [[buffer(0)]],device const long* idx [[buffer(1)]],device TYPE* out [[buffer(2)]], \
 constant ulong* xd [[buffer(3)]],constant ulong* xs [[buffer(4)]],constant uint& xr [[buffer(5)]],constant ulong& xo [[buffer(6)]], \
 constant ulong* id [[buffer(7)]],constant ulong* is [[buffer(8)]],constant uint& ir [[buffer(9)]],constant ulong& io [[buffer(10)]], \
 constant ulong* od [[buffer(11)]],constant uint& axis [[buffer(12)]],constant ulong& len [[buffer(13)]],uint gid [[thread_position_in_grid]]){ \
 if(gid>=len)return;long raw=idx[view_offset(logical_coord(gid,od,xr,axis),id,is,ir,io)];if(raw<0||ulong(raw)>=xd[axis]){out[gid]=TYPE(0);return;}ulong pick=ulong(raw);ulong base=xo; \
 for(uint a=0;a<xr;a++){ulong c=logical_coord(gid,od,xr,a);base+=(a==axis?pick:c)*xs[a];}out[gid]=x[base];} \
kernel void gather_##NAME(device const TYPE* x [[buffer(0)]],device const long* idx [[buffer(1)]],device TYPE* out [[buffer(2)]], \
 constant ulong* xd [[buffer(3)]],constant ulong* xs [[buffer(4)]],constant uint& xr [[buffer(5)]],constant ulong& xo [[buffer(6)]], \
 constant ulong* id [[buffer(7)]],constant ulong* is [[buffer(8)]],constant uint& ir [[buffer(9)]],constant ulong& io [[buffer(10)]], \
 constant uint& axis [[buffer(11)]],constant ulong& len [[buffer(12)]],uint gid [[thread_position_in_grid]]){if(gid>=len)return;long raw=idx[view_offset(gid,id,is,ir,io)];if(raw<0||ulong(raw)>=xd[axis]){out[gid]=TYPE(0);return;}ulong pick=ulong(raw),base=xo; \
 for(uint a=0;a<xr;a++){ulong c=logical_coord(gid,id,ir,a);base+=(a==axis?pick:c)*xs[a];}out[gid]=x[base];} \
kernel void index_add_##NAME(device const TYPE* x [[buffer(0)]],device const long* idx [[buffer(1)]],device const TYPE* src [[buffer(2)]],device TYPE* out [[buffer(3)]], \
 constant ulong* xd [[buffer(4)]],constant ulong* xs [[buffer(5)]],constant uint& xr [[buffer(6)]],constant ulong& xo [[buffer(7)]], \
 constant ulong* id [[buffer(8)]],constant ulong* is [[buffer(9)]],constant uint& ir [[buffer(10)]],constant ulong& io [[buffer(11)]], \
 constant ulong* sd [[buffer(12)]],constant ulong* ss [[buffer(13)]],constant uint& sr [[buffer(14)]],constant ulong& so [[buffer(15)]], \
 constant uint& axis [[buffer(16)]],constant ulong& out_len [[buffer(17)]],constant ulong& src_len [[buffer(18)]],uint gid [[thread_position_in_grid]]){if(gid>=out_len)return;ACC acc=TO_ACC(x[view_offset(gid,xd,xs,xr,xo)]); \
 for(ulong q=0;q<src_len;q++){bool same=true;for(uint a=0;a<xr;a++){ulong dc=logical_coord(gid,xd,xr,a),sc=logical_coord(q,sd,sr,a);if(a==axis){long raw=idx[view_offset(sc,id,is,ir,io)];if(raw<0||ulong(raw)>=xd[axis]){same=false;break;}sc=ulong(raw);}if(dc!=sc)same=false;}if(same)acc+=TO_ACC(src[view_offset(q,sd,ss,sr,so)]);}out[gid]=FROM_ACC(acc);} \
kernel void scatter_add_##NAME(device const TYPE* x [[buffer(0)]],device const long* idx [[buffer(1)]],device const TYPE* src [[buffer(2)]],device TYPE* out [[buffer(3)]], \
 constant ulong* xd [[buffer(4)]],constant ulong* xs [[buffer(5)]],constant uint& xr [[buffer(6)]],constant ulong& xo [[buffer(7)]], \
 constant ulong* id [[buffer(8)]],constant ulong* is [[buffer(9)]],constant uint& ir [[buffer(10)]],constant ulong& io [[buffer(11)]], \
 constant ulong* sd [[buffer(12)]],constant ulong* ss [[buffer(13)]],constant uint& sr [[buffer(14)]],constant ulong& so [[buffer(15)]], \
 constant uint& axis [[buffer(16)]],constant ulong& out_len [[buffer(17)]],constant ulong& src_len [[buffer(18)]],uint gid [[thread_position_in_grid]]){if(gid>=out_len)return;ACC acc=TO_ACC(x[view_offset(gid,xd,xs,xr,xo)]); \
 for(ulong q=0;q<src_len;q++){bool same=true;for(uint a=0;a<xr;a++){ulong dc=logical_coord(gid,xd,xr,a),sc=logical_coord(q,id,ir,a);if(a==axis){long raw=idx[view_offset(q,id,is,ir,io)];if(raw<0||ulong(raw)>=xd[axis]){same=false;break;}sc=ulong(raw);}if(dc!=sc)same=false;}if(same)acc+=TO_ACC(src[view_offset(q,sd,ss,sr,so)]);}out[gid]=FROM_ACC(acc);}
INDEX_KERNELS(half,float,f16,IDENTITY_F32,FROM_F16)
INDEX_KERNELS(float,float,f32,IDENTITY_F32,FROM_F32)
INDEX_KERNELS(long,long,i64,IDENTITY_I64,FROM_I64)

kernel void index_select_bool(device const uchar* x [[buffer(0)]],device const long* idx [[buffer(1)]],device uchar* out [[buffer(2)]],
 constant ulong* xd [[buffer(3)]],constant ulong* xs [[buffer(4)]],constant uint& xr [[buffer(5)]],constant ulong& xo [[buffer(6)]],
 constant ulong* id [[buffer(7)]],constant ulong* is [[buffer(8)]],constant uint& ir [[buffer(9)]],constant ulong& io [[buffer(10)]],
 constant ulong* od [[buffer(11)]],constant uint& axis [[buffer(12)]],constant ulong& len [[buffer(13)]],uint gid [[thread_position_in_grid]]){
 if(gid>=len)return;long raw=idx[view_offset(logical_coord(gid,od,xr,axis),id,is,ir,io)];if(raw<0||ulong(raw)>=xd[axis]){out[gid]=0;return;}ulong pick=ulong(raw),base=xo;
 for(uint a=0;a<xr;a++){ulong c=logical_coord(gid,od,xr,a);base+=(a==axis?pick:c)*xs[a];}out[gid]=x[base];}
kernel void gather_bool(device const uchar* x [[buffer(0)]],device const long* idx [[buffer(1)]],device uchar* out [[buffer(2)]],
 constant ulong* xd [[buffer(3)]],constant ulong* xs [[buffer(4)]],constant uint& xr [[buffer(5)]],constant ulong& xo [[buffer(6)]],
 constant ulong* id [[buffer(7)]],constant ulong* is [[buffer(8)]],constant uint& ir [[buffer(9)]],constant ulong& io [[buffer(10)]],
 constant uint& axis [[buffer(11)]],constant ulong& len [[buffer(12)]],uint gid [[thread_position_in_grid]]){if(gid>=len)return;long raw=idx[view_offset(gid,id,is,ir,io)];if(raw<0||ulong(raw)>=xd[axis]){out[gid]=0;return;}ulong pick=ulong(raw),base=xo;
 for(uint a=0;a<xr;a++){ulong c=logical_coord(gid,id,ir,a);base+=(a==axis?pick:c)*xs[a];}out[gid]=x[base];}

struct ConvParams { ulong n, ci, h, w, co, kh, kw, oh, ow, sh, sw, ph, pw, dh, dw; };
static bool source_pos(ulong out,ulong window,ulong stride,ulong dilation,ulong pad,ulong size,thread ulong& pos){ulong raw=out*stride+window*dilation;if(raw<pad)return false;pos=raw-pad;return pos<size;}

#define CONV_KERNELS(TYPE, ACC, NAME, TO_ACC, FROM_ACC) \
kernel void conv2d_##NAME(device const TYPE* x [[buffer(0)]],device const TYPE* w [[buffer(1)]],device TYPE* out [[buffer(2)]],constant ulong* xs [[buffer(3)]],constant ulong& xo [[buffer(4)]],constant ulong* ws [[buffer(5)]],constant ulong& wo [[buffer(6)]],constant ConvParams& p [[buffer(7)]],constant ulong& len [[buffer(8)]],uint gid [[thread_position_in_grid]]){if(gid>=len)return;ulong ow=gid%p.ow,t=gid/p.ow,oh=t%p.oh;t/=p.oh;ulong oc=t%p.co,b=t/p.co;ACC acc=ACC(0);for(ulong ic=0;ic<p.ci;ic++)for(ulong kh=0;kh<p.kh;kh++){ulong ih;if(!source_pos(oh,kh,p.sh,p.dh,p.ph,p.h,ih))continue;for(ulong kw=0;kw<p.kw;kw++){ulong iw;if(source_pos(ow,kw,p.sw,p.dw,p.pw,p.w,iw))acc+=TO_ACC(x[xo+b*xs[0]+ic*xs[1]+ih*xs[2]+iw*xs[3]])*TO_ACC(w[wo+oc*ws[0]+ic*ws[1]+kh*ws[2]+kw*ws[3]]);}}out[gid]=FROM_ACC(acc);} \
kernel void pool_##NAME(device const TYPE* x [[buffer(0)]],device TYPE* out [[buffer(1)]],constant ulong* xs [[buffer(2)]],constant ulong& xo [[buffer(3)]],constant ConvParams& p [[buffer(4)]],constant ulong& len [[buffer(5)]],constant uint& op [[buffer(6)]],uint gid [[thread_position_in_grid]]){if(gid>=len)return;ulong ow=gid%p.ow,t=gid/p.ow,oh=t%p.oh;t/=p.oh;ulong c=t%p.ci,b=t/p.ci;ACC acc=ACC(0);bool first=true;for(ulong kh=0;kh<p.kh;kh++){ulong ih;if(!source_pos(oh,kh,p.sh,1,p.ph,p.h,ih))continue;for(ulong kw=0;kw<p.kw;kw++){ulong iw;if(!source_pos(ow,kw,p.sw,1,p.pw,p.w,iw))continue;ACC v=TO_ACC(x[xo+b*xs[0]+c*xs[1]+ih*xs[2]+iw*xs[3]]);if(op==0){if(first||(!value_nan(acc)&&(value_nan(v)||v>acc)))acc=v;}else acc+=v;first=false;}}if(op==1)acc/=ACC(p.kh*p.kw);out[gid]=FROM_ACC(acc);} \
kernel void conv_input_grad_##NAME(device const TYPE* g [[buffer(0)]],device const TYPE* w [[buffer(1)]],device TYPE* out [[buffer(2)]],constant ulong* gs [[buffer(3)]],constant ulong& go [[buffer(4)]],constant ulong* ws [[buffer(5)]],constant ulong& wo [[buffer(6)]],constant ConvParams& p [[buffer(7)]],constant ulong& len [[buffer(8)]],uint gid [[thread_position_in_grid]]){if(gid>=len)return;ulong iw=gid%p.w,t=gid/p.w,ih=t%p.h;t/=p.h;ulong ic=t%p.ci,b=t/p.ci;ACC acc=ACC(0);for(ulong oc=0;oc<p.co;oc++)for(ulong oh=0;oh<p.oh;oh++)for(ulong ow=0;ow<p.ow;ow++)for(ulong kh=0;kh<p.kh;kh++){ulong sh;if(!source_pos(oh,kh,p.sh,p.dh,p.ph,p.h,sh)||sh!=ih)continue;for(ulong kw=0;kw<p.kw;kw++){ulong sw;if(source_pos(ow,kw,p.sw,p.dw,p.pw,p.w,sw)&&sw==iw)acc+=TO_ACC(g[go+b*gs[0]+oc*gs[1]+oh*gs[2]+ow*gs[3]])*TO_ACC(w[wo+oc*ws[0]+ic*ws[1]+kh*ws[2]+kw*ws[3]]);}}out[gid]=FROM_ACC(acc);} \
kernel void conv_weight_grad_##NAME(device const TYPE* g [[buffer(0)]],device const TYPE* x [[buffer(1)]],device TYPE* out [[buffer(2)]],constant ulong* gs [[buffer(3)]],constant ulong& go [[buffer(4)]],constant ulong* xs [[buffer(5)]],constant ulong& xo [[buffer(6)]],constant ConvParams& p [[buffer(7)]],constant ulong& len [[buffer(8)]],uint gid [[thread_position_in_grid]]){if(gid>=len)return;ulong kw=gid%p.kw,t=gid/p.kw,kh=t%p.kh;t/=p.kh;ulong ic=t%p.ci,oc=t/p.ci;ACC acc=ACC(0);for(ulong b=0;b<p.n;b++)for(ulong oh=0;oh<p.oh;oh++){ulong ih;if(!source_pos(oh,kh,p.sh,p.dh,p.ph,p.h,ih))continue;for(ulong ow=0;ow<p.ow;ow++){ulong iw;if(source_pos(ow,kw,p.sw,p.dw,p.pw,p.w,iw))acc+=TO_ACC(g[go+b*gs[0]+oc*gs[1]+oh*gs[2]+ow*gs[3]])*TO_ACC(x[xo+b*xs[0]+ic*xs[1]+ih*xs[2]+iw*xs[3]]);}}out[gid]=FROM_ACC(acc);} \
kernel void pool_backward_##NAME(device const TYPE* g [[buffer(0)]],device const TYPE* x [[buffer(1)]],device TYPE* out [[buffer(2)]],constant ulong* gs [[buffer(3)]],constant ulong& go [[buffer(4)]],constant ulong* xs [[buffer(5)]],constant ulong& xo [[buffer(6)]],constant ConvParams& p [[buffer(7)]],constant ulong& len [[buffer(8)]],constant uint& op [[buffer(9)]],uint gid [[thread_position_in_grid]]){if(gid>=len)return;ulong iw=gid%p.w,t=gid/p.w,ih=t%p.h;t/=p.h;ulong c=t%p.ci,b=t/p.ci;ACC acc=ACC(0);for(ulong oh=0;oh<p.oh;oh++)for(ulong ow=0;ow<p.ow;ow++){bool owns=op==1;ACC peak=ACC(0);ulong besth=0,bestw=0;bool first=true;for(ulong kh=0;kh<p.kh;kh++){ulong sh;if(!source_pos(oh,kh,p.sh,1,p.ph,p.h,sh))continue;for(ulong kw=0;kw<p.kw;kw++){ulong sw;if(!source_pos(ow,kw,p.sw,1,p.pw,p.w,sw))continue;ACC v=TO_ACC(x[xo+b*xs[0]+c*xs[1]+sh*xs[2]+sw*xs[3]]);if(first||(!value_nan(peak)&&(value_nan(v)||v>peak))){peak=v;besth=sh;bestw=sw;}first=false;}}if(op==0)owns=besth==ih&&bestw==iw;else owns=owns&&ih+p.ph>=oh*p.sh&&ih+p.ph<oh*p.sh+p.kh&&iw+p.pw>=ow*p.sw&&iw+p.pw<ow*p.sw+p.kw;if(owns){ACC v=TO_ACC(g[go+b*gs[0]+c*gs[1]+oh*gs[2]+ow*gs[3]]);acc+=op==1?v/ACC(p.kh*p.kw):v;}}out[gid]=FROM_ACC(acc);}
CONV_KERNELS(half,float,f16,IDENTITY_F32,FROM_F16)
CONV_KERNELS(float,float,f32,IDENTITY_F32,FROM_F32)
CONV_KERNELS(long,long,i64,IDENTITY_I64,FROM_I64)

#define FUSED_FLOAT(TYPE, NAME, FROM) \
kernel void softmax_##NAME(device const TYPE* x [[buffer(0)]],device TYPE* out [[buffer(1)]],constant ulong* d [[buffer(2)]],constant ulong* s [[buffer(3)]],constant uint& r [[buffer(4)]],constant ulong& off [[buffer(5)]],constant ulong& rows [[buffer(6)]],constant ulong& width [[buffer(7)]],uint row [[thread_position_in_grid]]){if(row>=rows)return;ulong base=reduced_offset(row,d,s,r,r-1,off);float peak=-INFINITY;bool nan=false;for(ulong c=0;c<width;c++){float v=float(x[base+c*s[r-1]]);nan|=isnan(v);peak=max(peak,v);}if(nan){for(ulong c=0;c<width;c++)out[row*width+c]=FROM(NAN);return;}if(peak==-INFINITY){for(ulong c=0;c<width;c++)out[row*width+c]=FROM(0);return;}float sum=0;for(ulong c=0;c<width;c++)sum+=exp(float(x[base+c*s[r-1]])-peak);for(ulong c=0;c<width;c++)out[row*width+c]=FROM(exp(float(x[base+c*s[r-1]])-peak)/sum);} \
kernel void layer_norm_##NAME(device const TYPE* x [[buffer(0)]],device const TYPE* w [[buffer(1)]],device const TYPE* b [[buffer(2)]],device TYPE* out [[buffer(3)]],device float* xhat [[buffer(4)]],device float* invout [[buffer(5)]],constant ulong* xd [[buffer(6)]],constant ulong* xs [[buffer(7)]],constant uint& xr [[buffer(8)]],constant ulong& xo [[buffer(9)]],constant ulong* ws [[buffer(10)]],constant ulong& woff [[buffer(11)]],constant ulong* bs [[buffer(12)]],constant ulong& boff [[buffer(13)]],constant ulong& rows [[buffer(14)]],constant ulong& width [[buffer(15)]],constant float& eps [[buffer(16)]],constant uint& save [[buffer(17)]],uint row [[thread_position_in_grid]]){if(row>=rows)return;ulong base=reduced_offset(row,xd,xs,xr,xr-1,xo);float mean=0;for(ulong c=0;c<width;c++)mean+=float(x[base+c*xs[xr-1]]);mean/=float(width);float var=0;for(ulong c=0;c<width;c++){float z=float(x[base+c*xs[xr-1]])-mean;var+=z*z;}float inv=rsqrt(var/float(width)+eps);if(save)invout[row]=inv;for(ulong c=0;c<width;c++){float z=(float(x[base+c*xs[xr-1]])-mean)*inv;if(save)xhat[row*width+c]=z;out[row*width+c]=FROM(z*float(w[woff+c*ws[0]])+float(b[boff+c*bs[0]]));}} \
kernel void layer_norm_backward_##NAME(device const TYPE* g [[buffer(0)]],device const float* xhat [[buffer(1)]],device const float* inv [[buffer(2)]],device const TYPE* w [[buffer(3)]],device TYPE* out [[buffer(4)]],constant ulong* gd [[buffer(5)]],constant ulong* gs [[buffer(6)]],constant uint& gr [[buffer(7)]],constant ulong& go [[buffer(8)]],constant ulong* hd [[buffer(9)]],constant ulong* hs [[buffer(10)]],constant uint& hr [[buffer(11)]],constant ulong& ho [[buffer(12)]],constant ulong* is [[buffer(13)]],constant ulong& io [[buffer(14)]],constant ulong* ws [[buffer(15)]],constant ulong& wo [[buffer(16)]],constant ulong& rows [[buffer(17)]],constant ulong& width [[buffer(18)]],uint row [[thread_position_in_grid]]){if(row>=rows)return;float sum=0,sumh=0;for(ulong c=0;c<width;c++){float dy=float(g[view_offset(row*width+c,gd,gs,gr,go)])*float(w[wo+c*ws[0]]),h=xhat[view_offset(row*width+c,hd,hs,hr,ho)];sum+=dy;sumh+=dy*h;}float iv=inv[io+row*is[0]];for(ulong c=0;c<width;c++){float dy=float(g[view_offset(row*width+c,gd,gs,gr,go)])*float(w[wo+c*ws[0]]),h=xhat[view_offset(row*width+c,hd,hs,hr,ho)];out[row*width+c]=FROM(iv*(dy-(sum+h*sumh)/float(width)));}}
FUSED_FLOAT(half,f16,FROM_F16)
FUSED_FLOAT(float,f32,FROM_F32)

#define OPT_KERNELS(TYPE, NAME, FROM) \
kernel void sgd_##NAME(device const TYPE* p [[buffer(0)]],device const TYPE* g [[buffer(1)]],device const float* vin [[buffer(2)]],device TYPE* pout [[buffer(3)]],device float* vout [[buffer(4)]],constant ulong& len [[buffer(5)]],constant float* hp [[buffer(6)]],constant uint& hasv [[buffer(7)]],constant uint& usem [[buffer(8)]],uint gid [[thread_position_in_grid]]){if(gid>=len)return;float pv=float(p[gid]),grad=float(g[gid])+hp[2]*pv,dir=hasv?hp[1]*vin[gid]+grad:grad;if(usem)vout[gid]=dir;pout[gid]=FROM(pv-hp[0]*dir);} \
kernel void adam_##NAME(device const TYPE* p [[buffer(0)]],device const TYPE* g [[buffer(1)]],device const float* mi [[buffer(2)]],device const float* vi [[buffer(3)]],device TYPE* po [[buffer(4)]],device float* mo [[buffer(5)]],device float* vo [[buffer(6)]],constant ulong& len [[buffer(7)]],constant float* h [[buffer(8)]],uint gid [[thread_position_in_grid]]){if(gid>=len)return;float pv=float(p[gid]),grad=float(g[gid]);bool dec=h[7]==1.0f;if(!dec)grad+=h[4]*pv;float m=h[1]*mi[gid]+(1-h[1])*grad,v=h[2]*vi[gid]+(1-h[2])*grad*grad,next=dec?pv*(1-h[0]*h[4]):pv;next-=h[0]*(m/h[5])/(sqrt(v/h[6])+h[3]);po[gid]=FROM(next);mo[gid]=m;vo[gid]=v;}
OPT_KERNELS(half,f16,FROM_F16)
OPT_KERNELS(float,f32,FROM_F32)
