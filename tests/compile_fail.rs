use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const BUILD_REQUIRED_MARKER: &str = "rstorch-ui: build";

#[test]
fn compile_fail_shape_guarantees() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cases = ui_cases(&manifest_dir);
    let mut check_cases = Vec::new();
    let mut build_cases = Vec::new();

    for source in cases {
        if source_requires_build(&source) {
            build_cases.push(source);
        } else {
            check_cases.push(source);
        }
    }

    let t = trybuild::TestCases::new();
    for source in check_cases {
        t.compile_fail(source);
    }

    run_build_required_cases(&manifest_dir, &build_cases);
}

fn run_build_required_cases(manifest_dir: &Path, cases: &[PathBuf]) {
    if cases.is_empty() {
        return;
    }

    let scratch = manifest_dir.join("target/compile-fail-build");
    let target_dir = scratch.join("target");
    if scratch.exists() {
        fs::remove_dir_all(&scratch).unwrap();
    }
    fs::create_dir_all(&scratch).unwrap();

    for source in cases {
        run_build_required_case(manifest_dir, &scratch, &target_dir, source);
    }
}

fn run_build_required_case(manifest_dir: &Path, scratch: &Path, target_dir: &Path, source: &Path) {
    let crate_name = crate_name_for(manifest_dir, source);
    let crate_dir = scratch.join(&crate_name);
    fs::create_dir_all(crate_dir.join("src")).unwrap();
    fs::write(
        crate_dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[dependencies]\nrstorch = {{ path = \"{}\"{} }}\n",
            toml_path(manifest_dir),
            dependency_features(&case_features(manifest_dir, source)),
        ),
    )
    .unwrap();
    fs::copy(source, crate_dir.join("src/main.rs")).unwrap();

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

    if overwrite_fixtures() {
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

fn ui_cases(manifest_dir: &Path) -> Vec<PathBuf> {
    let root = manifest_dir.join("tests/ui");
    let enabled = enabled_features();
    let mut cases = Vec::new();

    for entry in fs::read_dir(&root).unwrap() {
        let path = entry.unwrap().path();
        if !path.is_dir() {
            continue;
        }

        let Some(feature_set) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        if feature_set_active(feature_set, &enabled) {
            discover_rs_files(&path, &mut cases);
        }
    }

    cases.sort();
    cases
}

fn discover_rs_files(dir: &Path, cases: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            discover_rs_files(&path, cases);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            cases.push(path);
        }
    }
}

fn source_requires_build(source: &Path) -> bool {
    fs::read_to_string(source)
        .unwrap()
        .contains(BUILD_REQUIRED_MARKER)
}

fn feature_set_active(feature_set: &str, enabled: &[&'static str]) -> bool {
    if feature_set == "default" {
        return enabled.is_empty();
    }

    feature_set
        .split('_')
        .all(|feature| known_feature(feature) && enabled.contains(&feature))
}

fn enabled_features() -> Vec<&'static str> {
    let mut enabled = Vec::new();

    if cfg!(feature = "cuda") {
        enabled.push("cuda");
    }
    if cfg!(feature = "hub") {
        enabled.push("hub");
    }
    if cfg!(feature = "metal") {
        enabled.push("metal");
    }
    if cfg!(feature = "wgpu") {
        enabled.push("wgpu");
    }

    enabled
}

fn known_feature(feature: &str) -> bool {
    matches!(feature, "cuda" | "hub" | "metal" | "wgpu")
}

fn case_features(manifest_dir: &Path, source: &Path) -> Vec<String> {
    let root = manifest_dir.join("tests/ui");
    let relative = source.strip_prefix(&root).unwrap();
    let feature_set = relative
        .components()
        .next()
        .unwrap()
        .as_os_str()
        .to_string_lossy();

    if feature_set == "default" {
        Vec::new()
    } else {
        feature_set.split('_').map(str::to_owned).collect()
    }
}

fn crate_name_for(manifest_dir: &Path, source: &Path) -> String {
    let root = manifest_dir.join("tests/ui");
    let relative = source.strip_prefix(&root).unwrap().with_extension("");

    relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn toml_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

fn dependency_features(features: &[String]) -> String {
    if features.is_empty() {
        String::new()
    } else {
        let features = features
            .iter()
            .map(|feature| format!("\"{feature}\""))
            .collect::<Vec<_>>()
            .join(", ");
        format!(", features = [{features}]")
    }
}

fn normalize_stderr(stderr: &str, manifest_dir: &Path) -> String {
    let stderr = stderr.replace(&format!("{}/", manifest_dir.display()), "");
    let mut normalized = Vec::new();

    for line in stderr.lines() {
        if line.contains("full name for the type has been written to")
            || line.contains("consider using `--verbose` to print the full type name")
        {
            if normalized
                .last()
                .is_some_and(|line: &String| line.trim() == "|")
            {
                normalized.pop();
            }
            continue;
        }

        if line.contains("the above error was encountered while instantiating `") {
            normalized.push(
                "note: the above error was encountered while instantiating `<instantiation>`"
                    .to_string(),
            );
        } else {
            normalized.push(line.to_string());
        }
    }

    let mut stderr = normalized.join("\n");
    if stderr.is_empty() || stderr.ends_with('\n') {
        stderr
    } else {
        stderr.push('\n');
        stderr
    }
}

fn overwrite_fixtures() -> bool {
    std::env::var("TRYBUILD").as_deref() == Ok("overwrite")
}
