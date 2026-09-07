//! Pinned Rust 1.88 UI coverage for `#[derive(TypedModule)]`.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[test]
fn derive_typed_module_ui() {
    if std::env::var_os("RSTORCH_UI").is_none() {
        eprintln!(
            "skipping typed derive UI suite; set RSTORCH_UI=1 on the pinned toolchain to run it"
        );
        return;
    }

    let derive_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = derive_root
        .parent()
        .expect("derive crate is in the workspace");
    let cases_root = derive_root.join("tests/ui/typed");
    let run_root = derive_root
        .join("target/typed-ui")
        .join(std::process::id().to_string());
    if run_root.exists() {
        fs::remove_dir_all(&run_root).expect("stale typed derive UI directory is removable");
    }
    fs::create_dir_all(&run_root).expect("typed derive UI directory is creatable");

    let compiler = nested_rustc_version(&run_root);
    assert!(
        compiler.starts_with("rustc 1.88."),
        "typed derive UI fixtures require Rust 1.88; nested cargo uses {compiler}"
    );
    let overwrite = std::env::var_os("TRYBUILD").is_some_and(|value| value == "overwrite");

    let mut pass_cases = cases_in(&cases_root.join("pass"));
    let mut fail_cases = cases_in(&cases_root.join("fail"));
    pass_cases.sort();
    fail_cases.sort();
    assert!(!pass_cases.is_empty(), "typed derive UI needs pass cases");
    assert!(!fail_cases.is_empty(), "typed derive UI needs fail cases");

    let mut failures = Vec::new();
    for case in &pass_cases {
        if let Err(failure) = run_case(case, workspace, &derive_root, &run_root, true, overwrite) {
            failures.push(failure);
        }
    }
    for case in &fail_cases {
        if let Err(failure) = run_case(case, workspace, &derive_root, &run_root, false, overwrite) {
            failures.push(failure);
        }
    }

    fs::remove_dir_all(&run_root).expect("typed derive UI directory is removable");
    assert!(
        failures.is_empty(),
        "typed derive UI failures:\n\n{}",
        failures.join("\n\n")
    );
}

fn cases_in(directory: &Path) -> Vec<PathBuf> {
    fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to discover {}: {error}", directory.display()))
        .map(|entry| entry.expect("UI directory entry is readable").path())
        .filter(|path| path.extension() == Some(OsStr::new("rs")))
        .collect()
}

fn run_case(
    case: &Path,
    workspace: &Path,
    derive_root: &Path,
    run_root: &Path,
    should_pass: bool,
    overwrite: bool,
) -> Result<(), String> {
    let name = case
        .file_stem()
        .and_then(OsStr::to_str)
        .expect("UI case names are UTF-8")
        .replace('_', "-");
    let case_root = run_root.join(&name);
    fs::create_dir_all(&case_root).map_err(|error| error.to_string())?;
    // `{:?}` is deliberate here, not a `Display` oversight: it TOML-quotes
    // and escapes each path, which `Path::display()` would not.
    #[allow(clippy::unnecessary_debug_formatting)]
    let manifest = format!(
        "[package]\nname = \"rstorch-typed-derive-ui-{name}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[[bin]]\nname = \"case\"\npath = {case:?}\n\n[dependencies]\nrstorch = {{ path = {workspace:?}, features = [\"typed\"] }}\nrstorch-derive = {{ path = {derive_root:?} }}\n\n[workspace]\n"
    );
    fs::write(case_root.join("Cargo.toml"), manifest).map_err(|error| error.to_string())?;

    let output = Command::new("cargo")
        .arg("check")
        .arg("--quiet")
        .current_dir(&case_root)
        .env("CARGO_TERM_COLOR", "never")
        .env("CARGO_TARGET_DIR", run_root.join("target"))
        .output()
        .map_err(|error| format!("failed to run {}: {error}", case.display()))?;

    if should_pass {
        return if output.status.success() {
            Ok(())
        } else {
            Err(format!(
                "{} unexpectedly failed:\n{}",
                case.display(),
                normalized_stderr(&output, workspace, &case_root)
            ))
        };
    }
    if output.status.success() {
        return Err(format!("{} unexpectedly passed", case.display()));
    }

    let stderr = normalized_stderr(&output, workspace, &case_root);
    let expected_path = case.with_extension("stderr");
    if overwrite {
        fs::write(&expected_path, stderr).map_err(|error| error.to_string())?;
        return Ok(());
    }
    let expected = fs::read_to_string(&expected_path)
        .map_err(|error| format!("missing {}: {error}", expected_path.display()))?;
    if stderr == expected {
        Ok(())
    } else {
        Err(format!(
            "{} stderr mismatch\n--- expected\n{expected}--- actual\n{stderr}",
            case.display()
        ))
    }
}

fn normalized_stderr(output: &Output, workspace: &Path, case_root: &Path) -> String {
    String::from_utf8_lossy(&output.stderr)
        .replace('\\', "/")
        .replace(&case_root.to_string_lossy().replace('\\', "/"), "$CASE")
        .replace(
            &workspace.to_string_lossy().replace('\\', "/"),
            "$WORKSPACE",
        )
}

fn nested_rustc_version(run_root: &Path) -> String {
    let probe = run_root.join("compiler-probe");
    fs::create_dir_all(probe.join("src")).expect("compiler probe directory is creatable");
    fs::write(probe.join("src/lib.rs"), "").expect("compiler probe source is writable");
    fs::write(
        probe.join("Cargo.toml"),
        "[package]\nname = \"rstorch-typed-derive-ui-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[workspace]\n",
    )
    .expect("compiler probe manifest is writable");
    let output = Command::new("cargo")
        .arg("rustc")
        .arg("--quiet")
        .arg("--")
        .arg("--version")
        .current_dir(probe)
        .env("CARGO_TARGET_DIR", run_root.join("target"))
        .output()
        .expect("nested compiler probe executes");
    assert!(output.status.success(), "nested compiler probe succeeds");
    String::from_utf8(output.stdout)
        .expect("rustc version is UTF-8")
        .trim()
        .to_owned()
}
