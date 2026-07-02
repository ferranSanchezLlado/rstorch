use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn const_check_shape_guarantees() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let scratch = manifest_dir.join("target/const-check-compile-fail");
    let target_dir = scratch.join("target");
    if scratch.exists() {
        fs::remove_dir_all(&scratch).unwrap();
    }
    fs::create_dir_all(&scratch).unwrap();

    for source in const_check_cases(&manifest_dir) {
        run_const_check_case(&manifest_dir, &scratch, &target_dir, &source);
    }
}

fn const_check_cases(manifest_dir: &Path) -> Vec<PathBuf> {
    let mut cases = fs::read_dir(manifest_dir.join("tests/ui/const_check"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .collect::<Vec<_>>();
    cases.sort();
    cases
}

fn run_const_check_case(manifest_dir: &Path, scratch: &Path, target_dir: &Path, source: &Path) {
    let file_name = source.file_name().unwrap().to_string_lossy();
    let crate_name = file_name.trim_end_matches(".rs").replace('_', "-");
    let crate_dir = scratch.join(&crate_name);
    fs::create_dir_all(crate_dir.join("src")).unwrap();
    fs::write(
        crate_dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[dependencies]\nrstorch = {{ path = \"{}\" }}\n",
            toml_path(manifest_dir),
        ),
    )
    .unwrap();
    fs::copy(source, crate_dir.join("src/main.rs")).unwrap();

    // trybuild uses `cargo check`, but function-body const blocks only fire
    // when monomorphized. These cases need a real build.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .arg("build")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(crate_dir.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(target_dir)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "{} unexpectedly compiled successfully",
        source.display()
    );

    let actual = normalize_stderr(&String::from_utf8_lossy(&output.stderr), manifest_dir);
    let stderr = source.with_extension("stderr");

    if std::env::var("TRYBUILD").as_deref() == Ok("overwrite") {
        fs::write(stderr, actual).unwrap();
        return;
    }

    let expected = fs::read_to_string(&stderr).unwrap();
    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "{} stderr differed",
        source.display()
    );
}

fn toml_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

fn normalize_stderr(stderr: &str, manifest_dir: &Path) -> String {
    stderr.replace(&format!("{}/", manifest_dir.display()), "")
}
