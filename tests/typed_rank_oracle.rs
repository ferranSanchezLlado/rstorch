#![cfg(feature = "typed")]

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EXISTING_AXIS_CELLS: usize = 36;
const RANK_INCREASING_CELLS: usize = 36;
const TRANSPOSE_PAIRS: usize = 204;
const ASSOCIATED_OUTPUT_ASSERTIONS: usize =
    EXISTING_AXIS_CELLS * 7 + RANK_INCREASING_CELLS * 2 + TRANSPOSE_PAIRS;

#[test]
fn downstream_rank_axis_oracle_is_exhaustive_and_independent() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = workspace
        .join("target/typed-rank-oracle")
        .join(std::process::id().to_string());
    if root.exists() {
        fs::remove_dir_all(&root).expect("stale rank oracle directory must be removable");
    }
    fs::create_dir_all(root.join("src/bin")).expect("rank oracle directory must be creatable");
    fs::write(
        root.join("Cargo.toml"),
        format!(
            "[package]\nname = \"rstorch-typed-rank-oracle\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[dependencies]\nrstorch = {{ path = {workspace:?}, features = [\"typed\"] }}\n\n[workspace]\n"
        ),
    )
    .expect("rank oracle manifest must be writable");

    let (pass_source, assertions) = generated_pass_source();
    assert_eq!(assertions, ASSOCIATED_OUTPUT_ASSERTIONS);
    fs::write(root.join("src/bin/pass.rs"), pass_source)
        .expect("rank oracle pass source must be writable");
    fs::write(root.join("src/bin/ceiling.rs"), generated_ceiling_source())
        .expect("rank oracle ceiling source must be writable");

    let pass = cargo_check(&root, "pass");
    assert_success(&pass, "exhaustive rank oracle");

    let ceiling = cargo_check(&root, "ceiling");
    assert!(
        !ceiling.status.success(),
        "rank-8 increasing methods compiled"
    );
    let stderr = String::from_utf8_lossy(&ceiling.stderr);
    assert!(
        stderr.contains("unsqueeze"),
        "missing unsqueeze rejection: {stderr}"
    );
    assert!(
        stderr.contains("stack"),
        "missing stack rejection: {stderr}"
    );

    fs::remove_dir_all(&root).expect("rank oracle directory must be removable");
    eprintln!(
        "rank oracle checked {EXISTING_AXIS_CELLS} existing-axis cells, {RANK_INCREASING_CELLS} rank-increasing cells, {TRANSPOSE_PAIRS} transpose pairs, {ASSOCIATED_OUTPUT_ASSERTIONS} associated outputs, and 2 rank-ceiling rejections"
    );
}

fn generated_pass_source() -> (String, usize) {
    let mut source = String::from(
        "use rstorch::typed::{DYN, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, Tensor7, Tensor8};\n\
         use rstorch::typed::ops::{ArgKeepDimOutput, ArgOutput, ConcatOutput, IndexSelectOutput, InsertAxisOutput, KeepDimOutput, RemoveAxisOutput, ReplaceAxisOutput, StackOutput, TransposeOutput};\n\
         trait Same<T> {}\nimpl<T> Same<T> for T {}\nfn same<T: Same<U>, U>() {}\nfn main() {\n",
    );
    let mut assertions = 0;

    for rank in 1..=8 {
        let dims = marker_dims(rank);
        let input = tensor_type(&dims, None);
        for axis in 0..rank {
            let removed = without_axis(&dims, axis);
            let replaced = with_axis(&dims, axis, "99");
            let kept = with_axis(&dims, axis, "1");
            let selected = with_axis(&dims, axis, "23");
            let concatenated = with_axis(&dims, axis, "DYN");

            push_same(
                &mut source,
                &format!("<{input} as RemoveAxisOutput<{axis}>>::Output"),
                &tensor_type(&removed, None),
            );
            push_same(
                &mut source,
                &format!("<{input} as ReplaceAxisOutput<{axis}, 99>>::Output"),
                &tensor_type(&replaced, None),
            );
            push_same(
                &mut source,
                &format!("<{input} as KeepDimOutput<{axis}>>::Output"),
                &tensor_type(&kept, None),
            );
            push_same(
                &mut source,
                &format!("<{input} as ArgOutput<{axis}>>::Output"),
                &tensor_type(&removed, Some("i64")),
            );
            push_same(
                &mut source,
                &format!("<{input} as ArgKeepDimOutput<{axis}>>::Output"),
                &tensor_type(&kept, Some("i64")),
            );
            push_same(
                &mut source,
                &format!("<{input} as IndexSelectOutput<{axis}, Tensor1<23, i64>>>::Output"),
                &tensor_type(&selected, None),
            );
            push_same(
                &mut source,
                &format!("<{input} as ConcatOutput<{axis}>>::Output"),
                &tensor_type(&concatenated, None),
            );
            assertions += 7;
        }
    }

    for rank in 0..=7 {
        let dims = marker_dims(rank);
        let input = tensor_type(&dims, None);
        for axis in 0..=rank {
            let inserted = inserted_axis(&dims, axis, "99");
            let stacked = inserted_axis(&dims, axis, "DYN");
            push_same(
                &mut source,
                &format!("<{input} as InsertAxisOutput<{axis}, 99>>::Output"),
                &tensor_type(&inserted, None),
            );
            push_same(
                &mut source,
                &format!("<{input} as StackOutput<{axis}>>::Output"),
                &tensor_type(&stacked, None),
            );
            assertions += 2;
        }
    }

    for rank in 1..=8 {
        let dims = marker_dims(rank);
        let input = tensor_type(&dims, None);
        for left in 0..rank {
            for right in 0..rank {
                let mut output = dims.clone();
                output.swap(left, right);
                push_same(
                    &mut source,
                    &format!("<{input} as TransposeOutput<{left}, {right}>>::Output"),
                    &tensor_type(&output, None),
                );
                assertions += 1;
            }
        }
    }

    source.push_str("}\n");
    (source, assertions)
}

fn generated_ceiling_source() -> String {
    let rank8 = tensor_type(&marker_dims(8), None);
    format!(
        "use rstorch::typed::Tensor8;\nfn reject(value: &{rank8}) {{\n    let _ = value.unsqueeze::<8>();\n    let _ = {rank8}::stack::<8>(&[value]);\n}}\nfn main() {{}}\n"
    )
}

fn marker_dims(rank: usize) -> Vec<String> {
    (0..rank).map(|axis| (10 + axis).to_string()).collect()
}

fn without_axis(dims: &[String], axis: usize) -> Vec<String> {
    dims.iter()
        .enumerate()
        .filter(|(index, _)| *index != axis)
        .map(|(_, dim)| dim.clone())
        .collect()
}

fn with_axis(dims: &[String], axis: usize, value: &str) -> Vec<String> {
    let mut output = dims.to_vec();
    output[axis] = value.to_owned();
    output
}

fn inserted_axis(dims: &[String], axis: usize, value: &str) -> Vec<String> {
    let mut output = dims.to_vec();
    output.insert(axis, value.to_owned());
    output
}

fn tensor_type(dims: &[String], element: Option<&str>) -> String {
    let rank = dims.len();
    let mut arguments = dims.to_vec();
    if let Some(element) = element {
        arguments.push(element.to_owned());
    }
    if arguments.is_empty() {
        format!("Tensor{rank}")
    } else {
        format!("Tensor{rank}<{}>", arguments.join(", "))
    }
}

fn push_same(source: &mut String, actual: &str, expected: &str) {
    let _ = writeln!(source, "    same::<{actual}, {expected}>();");
}

fn cargo_check(root: &Path, binary: &str) -> Output {
    Command::new("cargo")
        .arg("check")
        .arg("--quiet")
        .arg("--offline")
        .arg("--bin")
        .arg(binary)
        .current_dir(root)
        .env(
            "CARGO_TARGET_DIR",
            Path::new(env!("CARGO_MANIFEST_DIR")).join("target/typed-rank-oracle/target"),
        )
        .output()
        .expect("rank oracle cargo check must execute")
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{context} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
