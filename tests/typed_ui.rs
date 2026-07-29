//! Compile-time and runtime guarantee harness for the opt-in typed API.
//!
//! Cases below `tests/ui/typed/` are ordinary `cargo check` failures unless
//! marked otherwise. `// rstorch-ui: build` monomorphizes body-level const
//! assertions, `// rstorch-ui: pass` requires check/build success, and
//! `// rstorch-ui: run` executes a runtime-deferred guarantee. Rust 1.88
//! failures match exactly; other toolchains assert adjacent semantic fragments.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BUILD_MARKER: &str = "// rstorch-ui: build";
const PASS_MARKER: &str = "// rstorch-ui: pass";
const RUN_MARKER: &str = "// rstorch-ui: run";

#[derive(Clone, Copy)]
enum Mode {
    FailCheck,
    FailBuild,
    PassCheck,
    PassBuild,
    Run,
}

impl Mode {
    fn command(self) -> &'static str {
        match self {
            Self::FailCheck | Self::PassCheck => "check",
            Self::FailBuild | Self::PassBuild => "build",
            Self::Run => "run",
        }
    }

    fn expects_success(self) -> bool {
        matches!(self, Self::PassCheck | Self::PassBuild | Self::Run)
    }

    fn index(self) -> usize {
        match self {
            Self::FailCheck => 0,
            Self::FailBuild => 1,
            Self::PassCheck => 2,
            Self::PassBuild => 3,
            Self::Run => 4,
        }
    }
}

#[test]
fn typed_compile_fail_ui() {
    if std::env::var_os("RSTORCH_UI").is_none() {
        eprintln!("skipping typed UI suite; set RSTORCH_UI=1 to run pinned or semantic checks");
        return;
    }

    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cases_root = workspace.join("tests/ui/typed");
    let mut cases = Vec::new();
    discover_cases(&cases_root, &mut cases);
    cases.sort();

    let modes = cases.iter().fold([0; 5], |mut modes, case| {
        let source = fs::read_to_string(case).expect("typed UI case must be readable");
        modes[case_mode(&source)
            .expect("typed UI case markers must be valid")
            .index()] += 1;
        modes
    });
    for (count, name) in modes.iter().zip([
        "failing check",
        "failing build",
        "passing check",
        "passing build",
        "passing run",
    ]) {
        assert!(*count > 0, "typed UI suite discovered no {name} cases");
    }

    let run_root = workspace
        .join("target/typed-ui")
        .join(std::process::id().to_string());
    if run_root.exists() {
        fs::remove_dir_all(&run_root).expect("stale typed UI run directory must be removable");
    }
    fs::create_dir_all(&run_root).expect("typed UI run directory must be creatable");

    // Probe through nested Cargo itself: this is the compiler that builds every
    // generated fixture, including when rustup selects it via RUSTUP_TOOLCHAIN.
    let nested_rustc = nested_rustc_version(&run_root);
    let pinned = nested_rustc.starts_with("rustc 1.88.");
    let overwrite = std::env::var_os("TRYBUILD").is_some_and(|value| value == "overwrite");
    assert!(
        !overwrite || pinned,
        "typed UI fixtures may only be overwritten with Rust 1.88; nested cargo uses {nested_rustc}"
    );
    eprintln!("typed UI nested compiler: {nested_rustc}");

    let mut failures = Vec::new();
    for case in &cases {
        if let Err(failure) = run_case(case, &workspace, &run_root, pinned, overwrite) {
            failures.push(failure);
        }
    }
    fs::remove_dir_all(&run_root).expect("typed UI run directory must be removable");

    assert!(
        failures.is_empty(),
        "typed UI failures:\n\n{}",
        failures.join("\n\n")
    );
    eprintln!(
        "typed UI executed {} failing checks, {} failing builds, {} passing checks, {} passing builds, and {} runtime cases",
        modes[0], modes[1], modes[2], modes[3], modes[4]
    );
}

fn case_mode(source: &str) -> Result<Mode, String> {
    let build = source.contains(BUILD_MARKER);
    let pass = source.contains(PASS_MARKER);
    let run = source.contains(RUN_MARKER);
    if run && (build || pass) {
        return Err("`rstorch-ui: run` cannot be combined with another mode marker".into());
    }
    Ok(match (build, pass, run) {
        (_, _, true) => Mode::Run,
        (true, true, false) => Mode::PassBuild,
        (false, true, false) => Mode::PassCheck,
        (true, false, false) => Mode::FailBuild,
        (false, false, false) => Mode::FailCheck,
    })
}

fn discover_cases(directory: &Path, cases: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to discover {}: {error}", directory.display()));
    for entry in entries {
        let path = entry
            .expect("typed UI directory entry must be readable")
            .path();
        if path.is_dir() {
            discover_cases(&path, cases);
        } else if path.extension() == Some(OsStr::new("rs")) {
            cases.push(path);
        }
    }
}

fn run_case(
    case: &Path,
    workspace: &Path,
    run_root: &Path,
    pinned: bool,
    overwrite: bool,
) -> Result<(), String> {
    let source = fs::read_to_string(case).map_err(|error| error.to_string())?;
    let mode = case_mode(&source)?;
    let case_name = case
        .file_stem()
        .and_then(OsStr::to_str)
        .expect("typed UI case names must be UTF-8")
        .replace('_', "-");
    let case_root = run_root.join(&case_name);
    fs::create_dir_all(case_root.join("src")).map_err(|error| error.to_string())?;
    fs::write(case_root.join("src/main.rs"), source).map_err(|error| error.to_string())?;
    fs::write(
        case_root.join("Cargo.toml"),
        format!(
            "[package]\nname = \"rstorch-typed-ui-{case_name}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[dependencies]\nrstorch = {{ path = {workspace:?}, features = [\"typed\"] }}\n\n[workspace]\n"
        ),
    )
    .map_err(|error| error.to_string())?;

    let output = Command::new("cargo")
        .arg(mode.command())
        .arg("--quiet")
        .current_dir(&case_root)
        .env("CARGO_TERM_COLOR", "never")
        .env("CARGO_TARGET_DIR", run_root.join("target"))
        .output()
        .map_err(|error| format!("failed to run cargo for {}: {error}", case.display()))?;
    if mode.expects_success() {
        if output.status.success() {
            return Ok(());
        }
        return Err(format!(
            "{} unexpectedly failed under cargo {}\n{}",
            relative_case(case, workspace).display(),
            mode.command(),
            normalized_stderr(&output, workspace, &case_root)
        ));
    }
    if output.status.success() {
        return Err(format!(
            "{} unexpectedly passed under cargo {}",
            relative_case(case, workspace).display(),
            mode.command()
        ));
    }

    let stderr = normalized_stderr(&output, workspace, &case_root);
    let expected_path = case.with_extension("stderr");
    if overwrite {
        fs::write(&expected_path, &stderr).map_err(|error| error.to_string())?;
    } else if pinned {
        let expected = fs::read_to_string(&expected_path).map_err(|error| {
            format!(
                "missing pinned fixture {}: {error}",
                expected_path.display()
            )
        })?;
        if stderr != expected {
            return Err(format!(
                "{} did not match its Rust 1.88 stderr fixture\n--- expected\n{expected}--- actual\n{stderr}",
                relative_case(case, workspace).display()
            ));
        }
    } else {
        assert_semantic_fragments(case, &stderr)?;
    }
    Ok(())
}

fn normalized_stderr(output: &Output, workspace: &Path, case_root: &Path) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).replace("\\", "/");
    let workspace = workspace.to_string_lossy().replace("\\", "/");
    let case_root = case_root.to_string_lossy().replace("\\", "/");
    stderr
        .replace(&case_root, "$CASE")
        .replace(&workspace, "$WORKSPACE")
}

fn assert_semantic_fragments(case: &Path, stderr: &str) -> Result<(), String> {
    let fragments_path = case.with_extension("stderr.fragments");
    let fragments = fs::read_to_string(&fragments_path).map_err(|error| {
        format!(
            "missing semantic fragments {}: {error}",
            fragments_path.display()
        )
    })?;
    for fragment in fragments.lines().filter(|line| !line.is_empty()) {
        if !stderr.contains(fragment) {
            return Err(format!(
                "{} is missing semantic diagnostic fragment `{fragment}`\n{stderr}",
                case.display()
            ));
        }
    }
    Ok(())
}

fn nested_rustc_version(run_root: &Path) -> String {
    let probe = run_root.join("compiler-probe");
    fs::create_dir_all(probe.join("src")).expect("compiler probe directory must be creatable");
    fs::write(probe.join("src/lib.rs"), "").expect("compiler probe source must be writable");
    fs::write(
        probe.join("Cargo.toml"),
        "[package]\nname = \"rstorch-ui-compiler-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[workspace]\n",
    )
    .expect("compiler probe manifest must be writable");
    let output = Command::new("cargo")
        .arg("rustc")
        .arg("--quiet")
        .arg("--")
        .arg("--version")
        .current_dir(probe)
        .env("CARGO_TARGET_DIR", run_root.join("target"))
        .output()
        .expect("nested cargo compiler probe must execute");
    assert!(
        output.status.success(),
        "nested cargo compiler probe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("nested rustc version must be UTF-8")
        .trim()
        .to_owned()
}

fn relative_case<'a>(case: &'a Path, workspace: &Path) -> &'a Path {
    case.strip_prefix(workspace).unwrap_or(case)
}
