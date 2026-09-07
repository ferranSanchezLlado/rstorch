enable f16;

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read> c: array<f16>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<storage, read> p: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;

fn address(logical:u32,base:u32)->u32{var rem=logical;var addr=p[base];var axis=p[base+1u];loop{if(axis==0u){break;}axis-=1u;let dim=p[base+2u+axis];addr+=(rem%dim)*p[base+10u+axis];rem/=dim;}return addr;}
fn address_fast(logical:u32,base:u32,contiguous:bool)->u32{if(contiguous){return p[base]+logical;}return address(logical,base);}
fn address_reduced(logical:u32,base:u32,axis:u32,value:u32)->u32{var rem=logical;var addr=p[base];var i=p[base+1u];loop{if(i==0u){break;}i-=1u;if(i==axis){addr+=value*p[base+10u+i];}else{let dim=p[base+2u+i];addr+=(rem%dim)*p[base+10u+i];rem/=dim;}}return addr;}

@compute @workgroup_size(256)
fn f16_compare(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=p[0]){return;}let x=a[address_fast(i,8u,p[5]!=0u)];let y=b[address_fast(i,26u,p[5]!=0u)];var yes=x==y;switch p[2]{case 1u:{yes=x!=y;}case 2u:{yes=x<y;}case 3u:{yes=x<=y;}case 4u:{yes=x>y;}case 5u:{yes=x>=y;}default:{}}out[i]=select(0u,1u,yes);}

@compute @workgroup_size(256)
fn f16_cast_out(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=p[0]){return;}let x=f32(a[address_fast(i,8u,p[3]!=0u)]);if(p[2]==0u){out[i]=bitcast<u32>(x);}else if(p[2]==1u){out[i]=select(0u,1u,x!=0.0);}else{let bits=bitcast<u32>(x);if((bits&0x7fffffffu)>0x7f800000u){out[i*2u]=0u;out[i*2u+1u]=0u;}else if(bits==0x7f800000u){out[i*2u]=0xffffffffu;out[i*2u+1u]=0x7fffffffu;}else if(bits==0xff800000u){out[i*2u]=0u;out[i*2u+1u]=0x80000000u;}else{let value=i32(x);out[i*2u]=bitcast<u32>(value);out[i*2u+1u]=select(0u,0xffffffffu,value<0);}}}

@compute @workgroup_size(256)
fn f16_arg_reduce(@builtin(global_invocation_id) gid:vec3<u32>){let oi=gid.x;if(oi>=p[0]){return;}let axis=p[3];let count=p[4];var best=a[address_reduced(oi,8u,axis,0u)];var best_i=0u;var k=1u;loop{if(k>=count){break;}let x=a[address_reduced(oi,8u,axis,k)];var take=false;if(best==best){take=x!=x;if(!take){take=select(x<best,x>best,p[2]==0u);}}if(take){best=x;best_i=k;}k+=1u;}out[oi*2u]=best_i;out[oi*2u+1u]=0u;}
