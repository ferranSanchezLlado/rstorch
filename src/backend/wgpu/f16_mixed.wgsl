enable f16;

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> c: array<f16>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
@group(0) @binding(4) var<storage, read> p: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;

fn address(logical: u32, base: u32) -> u32 {
    var rem=logical; var addr=p[base]; var axis=p[base+1u];
    loop { if(axis==0u){break;} axis-=1u; let dim=p[base+2u+axis]; addr+=(rem%dim)*p[base+10u+axis]; rem/=dim; }
    return addr;
}
fn address_fast(logical:u32,base:u32,contiguous:bool)->u32{if(contiguous){return p[base]+logical;}return address(logical,base);}
fn coord(logical:u32,base:u32,wanted:u32)->u32{var rem=logical;var axis=p[base+1u];loop{axis-=1u;let dim=p[base+2u+axis];let value=rem%dim;if(axis==wanted){return value;}rem/=dim;}return 0u;}
fn address_mapped(logical:u32,input_base:u32,logical_base:u32,axis:u32,value:u32)->u32{var rem=logical;var addr=p[input_base];var i=p[logical_base+1u];loop{if(i==0u){break;}i-=1u;let dim=p[logical_base+2u+i];var at=rem%dim;rem/=dim;if(i==axis){at=value;}addr+=at*p[input_base+10u+i];}return addr;}
fn logical_replacing(logical:u32,from_base:u32,to_base:u32,axis:u32,value:u32)->u32{var rem=logical;var result=0u;var multiplier=1u;var i=p[from_base+1u];loop{if(i==0u){break;}i-=1u;let dim=p[from_base+2u+i];var at=rem%dim;rem/=dim;if(i==axis){at=value;}result+=at*multiplier;multiplier*=p[to_base+2u+i];}return result;}
fn report_bad(lo:u32,hi:u32,axis:u32){loop{let claimed=atomicCompareExchangeWeak(&status[0],0u,1u);if(claimed.exchanged){atomicStore(&status[1],lo);atomicStore(&status[2],hi);atomicStore(&status[3],axis);return;}if(claimed.old_value!=0u){return;}}}
fn index_from_b(logical:u32,bound:u32,axis:u32)->u32{let at=address(logical,26u)*2u;let lo=b[at];let hi=b[at+1u];if(hi!=0u||lo>=bound){report_bad(lo,hi,axis);return 0u;}return lo;}

@compute @workgroup_size(256)
fn f16_validate_indices(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=p[1]){return;}let at=address(i,26u)*2u;let lo=b[at];let hi=b[at+1u];if(hi!=0u||lo>=p[3]){report_bad(lo,hi,p[2]);}}

@compute @workgroup_size(256)
fn f16_where(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<p[0]){out[i]=select(c[address_fast(i,44u,p[5]!=0u)],a[address_fast(i,8u,p[5]!=0u)],b[address_fast(i,26u,p[5]!=0u)]!=0u);}}

@compute @workgroup_size(256)
fn f16_masked_fill(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<p[0]){out[i]=select(a[address_fast(i,8u,p[5]!=0u)],f16(bitcast<f32>(p[2])),b[address_fast(i,26u,p[5]!=0u)]!=0u);}}

@compute @workgroup_size(256)
fn f16_index_select(@builtin(global_invocation_id) gid:vec3<u32>){let oi=gid.x;if(oi>=p[0]){return;}let axis=p[2];let j=coord(oi,44u,axis);out[oi]=a[address_mapped(oi,8u,44u,axis,index_from_b(j,p[3],axis))];}

@compute @workgroup_size(256)
fn f16_gather(@builtin(global_invocation_id) gid:vec3<u32>){let oi=gid.x;if(oi>=p[0]){return;}let axis=p[2];out[oi]=a[address_mapped(oi,8u,26u,axis,index_from_b(oi,p[3],axis))];}

@compute @workgroup_size(256)
fn f16_index_add(@builtin(global_invocation_id) gid:vec3<u32>){let oi=gid.x;if(oi>=p[0]){return;}let axis=p[2];let wanted=coord(oi,8u,axis);var sum=f32(a[address(oi,8u)]);var j=0u;loop{if(j>=p[4]){break;}if(index_from_b(j,p[3],axis)==wanted){sum+=f32(c[address_mapped(oi,44u,8u,axis,j)]);}j+=1u;}out[oi]=f16(sum);}

@compute @workgroup_size(256)
fn f16_scatter_add(@builtin(global_invocation_id) gid:vec3<u32>){let oi=gid.x;if(oi>=p[0]){return;}let axis=p[2];let wanted=coord(oi,8u,axis);var sum=f32(a[address(oi,8u)]);var j=0u;loop{if(j>=p[4]){break;}let si=logical_replacing(oi,8u,26u,axis,j);if(index_from_b(si,p[3],axis)==wanted){sum+=f32(c[address(si,44u)]);}j+=1u;}out[oi]=f16(sum);}
