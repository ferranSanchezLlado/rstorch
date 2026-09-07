fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // WSL supplies its Windows-backed CUDA driver here. Prefer it over any
    // native Linux driver package that may also provide libcuda.so.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some()
        && std::env::var_os("HOST") == std::env::var_os("TARGET")
        && std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() == Some(std::ffi::OsStr::new("linux"))
        && std::path::Path::new("/usr/lib/wsl/lib/libcuda.so.1").is_file()
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
    }
}
