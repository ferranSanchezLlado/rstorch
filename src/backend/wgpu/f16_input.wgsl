enable f16;

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> c: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
@group(0) @binding(4) var<storage, read> p: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;

fn address(logical:u32,base:u32)->u32{var rem=logical;var addr=p[base];var axis=p[base+1u];loop{if(axis==0u){break;}axis-=1u;let dim=p[base+2u+axis];addr+=(rem%dim)*p[base+10u+axis];rem/=dim;}return addr;}
fn address_fast(logical:u32,base:u32,contiguous:bool)->u32{if(contiguous){return p[base]+logical;}return address(logical,base);}

@compute @workgroup_size(256)
fn f16_cast_in(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=p[0]){return;}let at=address_fast(i,8u,p[3]!=0u);var value=0.0;if(p[2]==0u){value=bitcast<f32>(a[at]);}else if(p[2]==1u){value=select(0.0,1.0,a[at]!=0u);}else{let lo=a[at*2u];let hi=a[at*2u+1u];if((hi&0x80000000u)==0u){value=f32(hi)*4294967296.0+f32(lo);}else{let mag_lo=(~lo)+1u;let carry=select(0u,1u,mag_lo==0u);let mag_hi=(~hi)+carry;value=-(f32(mag_hi)*4294967296.0+f32(mag_lo));}}out[i]=f16(value);}
