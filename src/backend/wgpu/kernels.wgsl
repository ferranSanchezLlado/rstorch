// Portable kernels intentionally use u32 storage words. F32 is bitcast, Bool
// is 0/1, and I64 occupies two words so host values are never narrowed.
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> c: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<storage, read> p: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;

const MAX_RANK: u32 = 8u;

fn address(logical: u32, base: u32) -> u32 {
    var rem = logical;
    var addr = p[base];
    var axis = p[base + 1u];
    loop {
        if (axis == 0u) { break; }
        axis -= 1u;
        let dim = p[base + 2u + axis];
        let coord = rem % dim;
        rem /= dim;
        addr += coord * p[base + 10u + axis];
    }
    return addr;
}

fn address_fast(logical: u32, base: u32, contiguous: bool) -> u32 {
    if (contiguous) { return p[base] + logical; }
    return address(logical, base);
}

fn coord(logical: u32, base: u32, wanted: u32) -> u32 {
    var rem = logical;
    var axis = p[base + 1u];
    loop {
        axis -= 1u;
        let dim = p[base + 2u + axis];
        let value = rem % dim;
        if (axis == wanted) { return value; }
        rem /= dim;
    }
    return 0u;
}

fn address_replacing(logical: u32, base: u32, axis: u32, value: u32) -> u32 {
    var rem = logical;
    var addr = p[base];
    var i = p[base + 1u];
    loop {
        if (i == 0u) { break; }
        i -= 1u;
        let dim = p[base + 2u + i];
        var at = rem % dim;
        rem /= dim;
        if (i == axis) { at = value; }
        addr += at * p[base + 10u + i];
    }
    return addr;
}

fn address_reduced(logical: u32, base: u32, axis: u32, value: u32) -> u32 {
    var rem = logical;
    var addr = p[base];
    var i = p[base + 1u];
    loop {
        if (i == 0u) { break; }
        i -= 1u;
        if (i == axis) {
            addr += value * p[base + 10u + i];
        } else {
            let dim = p[base + 2u + i];
            addr += (rem % dim) * p[base + 10u + i];
            rem /= dim;
        }
    }
    return addr;
}

fn address_reduced_fast(logical: u32, base: u32, axis: u32, value: u32, width: u32, contiguous: bool) -> u32 {
    if (contiguous) { return p[base] + logical * width + value; }
    return address_reduced(logical, base, axis, value);
}

fn address_mapped(logical: u32, input_base: u32, logical_base: u32, axis: u32, value: u32) -> u32 {
    var rem = logical;
    var addr = p[input_base];
    var i = p[logical_base + 1u];
    loop {
        if (i == 0u) { break; }
        i -= 1u;
        let dim = p[logical_base + 2u + i];
        var at = rem % dim;
        rem /= dim;
        if (i == axis) { at = value; }
        addr += at * p[input_base + 10u + i];
    }
    return addr;
}

fn logical_replacing(logical: u32, from_base: u32, to_base: u32, axis: u32, value: u32) -> u32 {
    var rem = logical;
    var result = 0u;
    var multiplier = 1u;
    var i = p[from_base + 1u];
    loop {
        if (i == 0u) { break; }
        i -= 1u;
        let from_dim = p[from_base + 2u + i];
        var at = rem % from_dim;
        rem /= from_dim;
        if (i == axis) { at = value; }
        result += at * multiplier;
        multiplier *= p[to_base + 2u + i];
    }
    return result;
}

fn is_nan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn is_inf(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }
fn nan_max(x: f32, y: f32) -> f32 { return select(max(x, y), bitcast<f32>(0x7fc00000u), is_nan(x) || is_nan(y)); }
fn nan_min(x: f32, y: f32) -> f32 { return select(min(x, y), bitcast<f32>(0x7fc00000u), is_nan(x) || is_nan(y)); }
fn pool_better(candidate: f32, current: f32) -> bool {
    if (is_nan(candidate)) { return !is_nan(current); }
    return !is_nan(current) && candidate > current;
}
fn pool_equal(x: f32, y: f32) -> bool {
    return (is_nan(x) && is_nan(y)) || x == y;
}

fn erf_approx(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1.0 - poly * exp(-(z * z)));
}

@compute @workgroup_size(256)
fn copy_words(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p[0]) { return; }
    let words = p[2];
    let src = address_fast(i, 8u, p[3] != 0u) * words;
    let dst = address_fast(i, 26u, p[4] != 0u) * words;
    out[dst] = a[src];
    if (words == 2u) { out[dst + 1u] = a[src + 1u]; }
}

@compute @workgroup_size(256)
fn full(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p[0]) { return; }
    out[i * p[2]] = p[3];
    if (p[2] == 2u) { out[i * 2u + 1u] = p[4]; }
}

@compute @workgroup_size(256)
fn cast_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p[0]) { return; }
    let src = a[address_fast(i, 8u, p[3] != 0u)];
    if (p[2] == 0u) { out[i] = bitcast<u32>(select(0.0, 1.0, src != 0u)); }
    else { out[i] = select(0u, 1u, bitcast<f32>(src) != 0.0); }
}

@compute @workgroup_size(256)
fn binary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p[0]) { return; }
    let x = bitcast<f32>(a[address_fast(i, 8u, p[5] != 0u)]);
    let y = bitcast<f32>(b[address_fast(i, 26u, p[5] != 0u)]);
    var z = x + y;
    switch p[2] {
        case 1u: { z = x - y; }
        case 2u: { z = x * y; }
        case 3u: { z = x / y; }
        case 4u: { z = nan_max(x, y); }
        case 5u: { z = nan_min(x, y); }
        default: {}
    }
    out[i] = bitcast<u32>(z);
}

@compute @workgroup_size(256)
fn binary_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p[0]) { return; }
    let x = bitcast<f32>(a[address_fast(i, 8u, p[5] != 0u)]);
    let y = bitcast<f32>(p[3]);
    var z = x + y;
    switch p[2] {
        case 1u: { z = x - y; }
        case 2u: { z = x * y; }
        case 3u: { z = x / y; }
        case 4u: { z = nan_max(x, y); }
        case 5u: { z = nan_min(x, y); }
        default: {}
    }
    out[i] = bitcast<u32>(z);
}

@compute @workgroup_size(256)
fn unary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p[0]) { return; }
    let x = bitcast<f32>(a[address_fast(i, 8u, p[5] != 0u)]);
    var z = nan_max(x, 0.0);
    switch p[2] {
        case 1u: { z = 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476)); }
        case 2u: { z = exp(x); }
        case 3u: { z = log(x); }
        case 4u: { z = sqrt(x); }
        case 5u: { z = select(tanh(x), sign(x), is_inf(x)); }
        case 6u: { z = 1.0 / (1.0 + exp(-x)); }
        case 7u: { z = -x; }
        case 8u: { z = abs(x); }
        default: {}
    }
    out[i] = bitcast<u32>(z);
}

@compute @workgroup_size(256)
fn compare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p[0]) { return; }
    let x = bitcast<f32>(a[address_fast(i, 8u, p[5] != 0u)]);
    let y = bitcast<f32>(b[address_fast(i, 26u, p[5] != 0u)]);
    var yes = x == y;
    switch p[2] {
        case 1u: { yes = x != y; }
        case 2u: { yes = x < y; }
        case 3u: { yes = x <= y; }
        case 4u: { yes = x > y; }
        case 5u: { yes = x >= y; }
        default: {}
    }
    out[i] = select(0u, 1u, yes);
}

@compute @workgroup_size(256)
fn where_cond(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p[0]) { return; }
    out[i] = select(c[address_fast(i, 44u, p[5] != 0u)], b[address_fast(i, 26u, p[5] != 0u)], a[address_fast(i, 8u, p[5] != 0u)] != 0u);
}

@compute @workgroup_size(256)
fn masked_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p[0]) { return; }
    out[i] = select(a[address_fast(i, 8u, p[5] != 0u)], p[2], b[address_fast(i, 26u, p[5] != 0u)] != 0u);
}

var<workgroup> reduce_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let oi = group.x;
    if (oi >= p[0]) { return; }
    let axis = p[3];
    let count = p[4];
    var value = select(
        bitcast<f32>(0x7f800000u),
        bitcast<f32>(0xff800000u),
        p[2] == 2u,
    );
    if (p[2] <= 1u) { value = 0.0; }
    var k = local.x;
    loop {
        if (k >= count) { break; }
        let at = address_reduced_fast(oi, 8u, axis, k, count, p[5] != 0u);
        let x = bitcast<f32>(a[at]);
        switch p[2] {
            case 0u, 1u: { value += x; }
            case 2u: { value = nan_max(value, x); }
            case 3u: { value = nan_min(value, x); }
            default: {}
        }
        k += 256u;
    }
    reduce_shared[local.x] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (local.x < stride) {
            switch p[2] {
                case 0u, 1u: { reduce_shared[local.x] += reduce_shared[local.x + stride]; }
                case 2u: { reduce_shared[local.x] = nan_max(reduce_shared[local.x], reduce_shared[local.x + stride]); }
                case 3u: { reduce_shared[local.x] = nan_min(reduce_shared[local.x], reduce_shared[local.x + stride]); }
                default: {}
            }
        }
        workgroupBarrier();
        stride /= 2u;
    }
    if (local.x == 0u) {
        value = reduce_shared[0];
        if (p[2] == 1u) { value /= f32(count); }
        out[oi] = bitcast<u32>(value);
    }
}

@compute @workgroup_size(256)
fn arg_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oi = gid.x;
    if (oi >= p[0]) { return; }
    let axis = p[3];
    let count = p[4];
    var best = bitcast<f32>(a[address_reduced(oi, 8u, axis, 0u)]);
    var best_i = 0u;
    var k = 1u;
    loop {
        if (k >= count) { break; }
        let x = bitcast<f32>(a[address_reduced(oi, 8u, axis, k)]);
        var take = false;
        if (!is_nan(best)) {
            take = is_nan(x);
            if (!take) { take = select(x < best, x > best, p[2] == 0u); }
        }
        if (take) { best = x; best_i = k; }
        k += 1u;
    }
    out[oi * 2u] = best_i;
    out[oi * 2u + 1u] = 0u;
}

var<workgroup> softmax_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax(
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let row = group.x;
    if (row >= p[2]) { return; }
    let width = p[3];
    let start = row * width;
    var maximum = bitcast<f32>(0xff800000u);
    var k = local.x;
    loop {
        if (k >= width) { break; }
        maximum = nan_max(maximum, bitcast<f32>(a[address_fast(start + k, 8u, p[5] != 0u)]));
        k += 256u;
    }
    softmax_shared[local.x] = maximum;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (local.x < stride) {
            softmax_shared[local.x] = nan_max(
                softmax_shared[local.x],
                softmax_shared[local.x + stride],
            );
        }
        workgroupBarrier();
        stride /= 2u;
    }
    maximum = softmax_shared[0];
    if (is_inf(maximum) && maximum < 0.0) {
        k = local.x;
        loop {
            if (k >= width) { break; }
            out[start + k] = bitcast<u32>(0.0);
            k += 256u;
        }
        return;
    }
    var total = 0.0;
    k = local.x;
    loop {
        if (k >= width) { break; }
        let value = exp(bitcast<f32>(a[address_fast(start + k, 8u, p[5] != 0u)]) - maximum);
        out[start + k] = bitcast<u32>(value);
        total += value;
        k += 256u;
    }
    softmax_shared[local.x] = total;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (local.x < stride) {
            softmax_shared[local.x] += softmax_shared[local.x + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    storageBarrier();
    k = local.x;
    loop {
        if (k >= width) { break; }
        out[start + k] = bitcast<u32>(bitcast<f32>(out[start + k]) / softmax_shared[0]);
        k += 256u;
    }
}

var<workgroup> layer_norm_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn layer_norm(
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let row = group.x;
    if (row >= p[2]) { return; }
    let width = p[3];
    let start = row * width;
    var mean = 0.0;
    var k = local.x;
    loop {
        if (k >= width) { break; }
        mean += bitcast<f32>(a[address_fast(start + k, 8u, p[5] != 0u)]);
        k += 256u;
    }
    layer_norm_shared[local.x] = mean;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (local.x < stride) {
            layer_norm_shared[local.x] += layer_norm_shared[local.x + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    mean = layer_norm_shared[0] / f32(width);
    var variance = 0.0;
    k = local.x;
    loop {
        if (k >= width) { break; }
        let centered = bitcast<f32>(a[address_fast(start + k, 8u, p[5] != 0u)]) - mean;
        variance += centered * centered;
        k += 256u;
    }
    layer_norm_shared[local.x] = variance;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (local.x < stride) {
            layer_norm_shared[local.x] += layer_norm_shared[local.x + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    let inv_std = inverseSqrt(layer_norm_shared[0] / f32(width) + bitcast<f32>(p[4]));
    k = local.x;
    loop {
        if (k >= width) { break; }
        let normalized = (bitcast<f32>(a[address_fast(start + k, 8u, p[5] != 0u)]) - mean) * inv_std;
        let weight = bitcast<f32>(b[address_fast(k, 26u, p[5] != 0u)]);
        let bias = bitcast<f32>(c[address_fast(k, 44u, p[5] != 0u)]);
        out[start + k] = bitcast<u32>(normalized * weight + bias);
        k += 256u;
    }
}

fn batch_address(batch: u32, base: u32) -> u32 {
    let rank = p[base + 1u];
    var rem = batch;
    var addr = p[base];
    var axis = rank - 2u;
    loop {
        if (axis == 0u) { break; }
        axis -= 1u;
        let dim = p[base + 2u + axis];
        addr += (rem % dim) * p[base + 10u + axis];
        rem /= dim;
    }
    return addr;
}

@compute @workgroup_size(256)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oi = gid.x;
    if (oi >= p[0]) { return; }
    let m_size = p[2]; let n_size = p[3]; let k_size = p[4];
    let n = oi % n_size; let m = (oi / n_size) % m_size; let batch = oi / (m_size * n_size);
    let ar = p[9]; let br = p[27];
    let abase = batch_address(batch, 8u) + m * p[18u + ar - 2u];
    let bbase = batch_address(batch, 26u) + n * p[36u + br - 1u];
    var sum = 0.0; var k = 0u;
    loop {
        if (k >= k_size) { break; }
        sum += bitcast<f32>(a[abase + k * p[18u + ar - 1u]]) * bitcast<f32>(b[bbase + k * p[36u + br - 2u]]);
        k += 1u;
    }
    out[oi] = bitcast<u32>(sum);
}

fn report_bad(lo: u32, hi: u32, axis: u32) {
    if (atomicCompareExchangeWeak(&status[0], 0u, 1u).exchanged) {
        atomicStore(&status[1], lo); atomicStore(&status[2], hi); atomicStore(&status[3], axis);
    }
}

fn index_from_b(logical: u32, bound: u32, axis: u32) -> u32 {
    let at = address(logical, 26u) * 2u; let lo = b[at]; let hi = b[at + 1u];
    if (hi != 0u || lo >= bound) { report_bad(lo, hi, axis); return 0u; }
    return lo;
}

@compute @workgroup_size(256)
fn validate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= p[1]) { return; }
    let at = address(i, 26u) * 2u;
    let lo = b[at]; let hi = b[at + 1u];
    if (hi != 0u || lo >= p[3]) { report_bad(lo, hi, p[2]); }
}

@compute @workgroup_size(256)
fn index_select(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oi = gid.x; if (oi >= p[0]) { return; }
    let axis = p[2]; let j = coord(oi, 44u, axis);
    let selected = index_from_b(j, p[3], axis);
    out[oi] = a[address_mapped(oi, 8u, 44u, axis, selected)];
}

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oi = gid.x; if (oi >= p[0]) { return; }
    let axis = p[2]; let selected = index_from_b(oi, p[3], axis);
    out[oi] = a[address_mapped(oi, 8u, 26u, axis, selected)];
}

@compute @workgroup_size(256)
fn index_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oi = gid.x; if (oi >= p[0]) { return; }
    let axis = p[2]; let wanted = coord(oi, 8u, axis); var sum = bitcast<f32>(a[address(oi, 8u)]);
    var j = 0u;
    loop {
        if (j >= p[4]) { break; }
        let selected = index_from_b(j, p[3], axis);
        if (selected == wanted) { sum += bitcast<f32>(c[address_mapped(oi, 44u, 8u, axis, j)]); }
        j += 1u;
    }
    out[oi] = bitcast<u32>(sum);
}

@compute @workgroup_size(256)
fn scatter_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oi = gid.x; if (oi >= p[0]) { return; }
    let axis = p[2]; let wanted = coord(oi, 8u, axis); var sum = bitcast<f32>(a[address(oi, 8u)]);
    var j = 0u;
    loop {
        if (j >= p[4]) { break; }
        let si = logical_replacing(oi, 8u, 26u, axis, j);
        let selected = index_from_b(si, p[3], axis);
        if (selected == wanted) { sum += bitcast<f32>(c[address(si, 44u)]); }
        j += 1u;
    }
    out[oi] = bitcast<u32>(sum);
}

fn addr4(base: u32, n: u32, ch: u32, y: u32, x: u32) -> u32 {
    return p[base] + n*p[base+10u] + ch*p[base+11u] + y*p[base+12u] + x*p[base+13u];
}

// Geometry occupies descriptor slots unused by rank-4 operands.
@compute @workgroup_size(256)
fn conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oi = gid.x; if (oi >= p[0]) { return; }
    let op=p[2]; let n_size=p[3]; let ic=p[4]; let ih=p[5]; let iw=p[6]; let oc=p[7];
    let kh_size=p[14]; let kw_size=p[15]; let oh_size=p[16]; let ow_size=p[17];
    let sh=p[22]; let sw=p[23]; let ph=p[24]; let pw=p[25]; let dh=p[32]; let dw=p[33];
    if (op <= 2u) {
        let ow=oi%ow_size; let oh=(oi/ow_size)%oh_size; let ch=(oi/(ow_size*oh_size))%oc; let n=oi/(ow_size*oh_size*oc);
        var value=select(0.0, bitcast<f32>(0xff800000u), op==1u); var count=0u; var ci=0u;
        loop { if (ci>=select(ic,1u,op!=0u)){break;} var ky=0u;
            loop {if(ky>=kh_size){break;} var kx=0u;
                loop {if(kx>=kw_size){break;} let sy=oh*sh+ky*dh; let sx=ow*sw+kx*dw;
                    if(sy>=ph && sx>=pw && sy-ph<ih && sx-pw<iw){let av=bitcast<f32>(a[addr4(8u,n,select(ci,ch,op!=0u),sy-ph,sx-pw)]);
                        if(op==0u){value += av*bitcast<f32>(b[addr4(26u,ch,ci,ky,kx)]);} else if(op==1u){if(pool_better(av,value)){value=av;}} else {value+=av; count+=1u;}}
                    kx+=1u;} ky+=1u;} ci+=1u;}
        if(op==2u){value/=f32(kh_size*kw_size);} out[oi]=bitcast<u32>(value); return;
    }
    if (op == 3u) {
        let x=oi%iw; let y=(oi/iw)%ih; let ci=(oi/(iw*ih))%ic; let n=oi/(iw*ih*ic); var sum=0.0; var co=0u;
        loop{if(co>=oc){break;} var oh=0u; loop{if(oh>=oh_size){break;} var ow=0u; loop{if(ow>=ow_size){break;} var ky=0u;
            loop{if(ky>=kh_size){break;} var kx=0u; loop{if(kx>=kw_size){break;} let sy=oh*sh+ky*dh; let sx=ow*sw+kx*dw;
                if(sy>=ph&&sx>=pw&&sy-ph==y&&sx-pw==x){sum+=bitcast<f32>(a[addr4(8u,n,co,oh,ow)])*bitcast<f32>(b[addr4(26u,co,ci,ky,kx)]);} kx+=1u;} ky+=1u;} ow+=1u;} oh+=1u;} co+=1u;}
        out[oi]=bitcast<u32>(sum); return;
    }
    if (op == 4u) {
        let kx=oi%kw_size; let ky=(oi/kw_size)%kh_size; let ci=(oi/(kw_size*kh_size))%ic; let co=oi/(kw_size*kh_size*ic); var sum=0.0; var n=0u;
        loop{if(n>=n_size){break;} var oh=0u; loop{if(oh>=oh_size){break;} var ow=0u; loop{if(ow>=ow_size){break;} let sy=oh*sh+ky*dh; let sx=ow*sw+kx*dw;
            if(sy>=ph&&sx>=pw&&sy-ph<ih&&sx-pw<iw){sum+=bitcast<f32>(a[addr4(8u,n,co,oh,ow)])*bitcast<f32>(b[addr4(26u,n,ci,sy-ph,sx-pw)]);} ow+=1u;} oh+=1u;} n+=1u;}
        out[oi]=bitcast<u32>(sum); return;
    }
    let x=oi%iw; let y=(oi/iw)%ih; let ch=(oi/(iw*ih))%ic; let n=oi/(iw*ih*ic); var sum=0.0; var oh=0u;
    loop{if(oh>=oh_size){break;} var ow=0u; loop{if(ow>=ow_size){break;} var inside=false; var is_max=true; var before=true; let grad=bitcast<f32>(a[addr4(8u,n,ch,oh,ow)]); var ky=0u;
        loop{if(ky>=kh_size){break;} var kx=0u; loop{if(kx>=kw_size){break;} let sy=oh*sh+ky; let sx=ow*sw+kx;
             if(sy>=ph&&sx>=pw&&sy-ph<ih&&sx-pw<iw){let candidate=sy-ph==y&&sx-pw==x; let other=bitcast<f32>(b[addr4(26u,n,ch,sy-ph,sx-pw)]); let current=bitcast<f32>(b[addr4(26u,n,ch,y,x)]); if(candidate){inside=true;before=false;} else if(op==5u && (pool_better(other,current) || (before && pool_equal(other,current)))){is_max=false;}} kx+=1u;} ky+=1u;}
        if(inside){if(op==6u){sum+=grad/f32(kh_size*kw_size);}else if(is_max){sum+=grad;}} ow+=1u;} oh+=1u;}
    out[oi]=bitcast<u32>(sum);
}
