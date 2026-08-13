//! Optimizer-state persistence through the versioned
//! [`Envelope`](crate::persist::Envelope).
//!
//! An optimizer's state is two things: a handful of scalar hyperparameters plus
//! per-parameter **step clocks**, and a moment buffer or two per parameter. The
//! envelope already carries exactly those two shapes — an opaque string section
//! for non-tensor state and ordinary safetensors tensors — so the encoding here
//! is deliberately small:
//!
//! - tensors go in as `optim.<dotted path>.<buffer name>` (e.g.
//!   `optim.fc1.weight.m`), keeping the `optim.` namespace clear of the model's
//!   own state-dict keys so one envelope can hold both;
//! - the `optimizer` section is a line-oriented `key=value` text:
//!
//! ```text
//! version=1
//! kind=adam
//! steps=7
//! hyper.lr=0.001
//! clock.fc1.weight=7
//! ```
//!
//! Reads are strict: an unknown key, an unparseable number, a wrong `kind` or a
//! stale `version` is an [`Error::Persistence`](crate::Error::Persistence)
//! rather than a partially-understood optimizer. Floats are written with `{:?}`,
//! which is the shortest representation that parses back exactly, so a resumed
//! run has bit-identical hyperparameters.
//!
//! **Parameter groups are code, not state.** The predicates and their overrides
//! are closures; a resumed run reconstructs them by building the optimizer the
//! same way. Only the base hyperparameters, the clocks and the moments persist.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use crate::autograd::GradKey;
use crate::error::{Error, Result};
use crate::nn::Module;
use crate::persist::{Envelope, HostTensor};
use crate::tensor::Tensor;

use super::engine::{self, Decode, Groups, ParamState, Rule, States};
use crate::checkpoint::{from_host_tensor, to_host_tensor};

/// The envelope section name (the one `persist` documents for this purpose).
const SECTION: &str = "optimizer";
/// Reserved tensor-key namespace for optimizer buffers.
const TENSOR_PREFIX: &str = "optim.";
/// Encoding version of the section text (independent of the envelope's own
/// format version, which covers the file, not this section's grammar).
const ENCODING_VERSION: u32 = 1;

/// One parameter's state on the way out: its path, its step clock, and its
/// named moment buffers (`velocity`, or `m`/`v`).
pub(crate) struct OutgoingParam<'a> {
    pub(crate) path: &'a str,
    pub(crate) clock: u64,
    pub(crate) buffers: Vec<(&'static str, &'a Tensor)>,
}

/// Write an optimizer's state into `envelope`.
///
/// Rejects an envelope that already carries optimizer state, so two optimizers
/// (or two saves of one) can never silently interleave their buffers.
///
/// Every buffer is copied to the host and every path checked **before** the
/// envelope is touched, so a rejected save leaves the envelope exactly as it
/// was — the caller can fix the problem and save the same envelope again
/// (whereas a half-written one would then trip the "already carries optimizer
/// state" guard above and look like a different bug).
fn save(
    envelope: &mut Envelope,
    kind: &'static str,
    hypers: &[(&'static str, f64)],
    steps: u64,
    params: &[OutgoingParam<'_>],
) -> Result<()> {
    if envelope.section(SECTION).is_some()
        || envelope
            .tensors()
            .keys()
            .any(|key| key.starts_with(TENSOR_PREFIX))
    {
        return Err(Error::Persistence {
            msg: format!(
                "envelope already carries optimizer state (a `{SECTION}` section or an \
                 `{TENSOR_PREFIX}*` tensor); save each optimizer into its own envelope"
            ),
        });
    }

    let mut text = String::new();
    let _ = writeln!(text, "version={ENCODING_VERSION}");
    let _ = writeln!(text, "kind={kind}");
    let _ = writeln!(text, "steps={steps}");
    let mut section_keys = std::collections::BTreeSet::from([
        "version".to_string(),
        "kind".to_string(),
        "steps".to_string(),
    ]);
    for (name, value) in hypers {
        let key = format!("hyper.{name}");
        if !section_keys.insert(key.clone()) {
            return Err(Error::Persistence {
                msg: format!("duplicate optimizer state key `{key}`"),
            });
        }
        let _ = writeln!(text, "hyper.{name}={value:?}");
    }
    let mut staged: Vec<(String, HostTensor)> = Vec::new();
    let mut buffer_keys = std::collections::BTreeSet::new();
    for param in params {
        // Paths become part of a `key=value` line and of a tensor key, so the
        // two characters that would make the encoding ambiguous are refused
        // rather than silently mangled.
        if param.path.contains('=') || param.path.contains('\n') {
            return Err(Error::Persistence {
                msg: format!(
                    "parameter path {:?} contains `=` or a newline, which the optimizer \
                     state encoding cannot represent",
                    param.path
                ),
            });
        }
        let clock_key = format!("clock.{}", param.path);
        if !section_keys.insert(clock_key.clone()) {
            return Err(Error::Persistence {
                msg: format!("duplicate optimizer state key `{clock_key}`"),
            });
        }
        let _ = writeln!(text, "clock.{}={}", param.path, param.clock);
        for (name, tensor) in &param.buffers {
            let key = format!("{TENSOR_PREFIX}{}.{name}", param.path);
            if !buffer_keys.insert(key.clone()) {
                return Err(Error::Persistence {
                    msg: format!("duplicate optimizer buffer `{key}`"),
                });
            }
            staged.push((key, to_host_tensor(tensor)?));
        }
    }

    // Past here nothing can fail but `set_section`, whose only rejection is a
    // section name — and `SECTION` is a constant with no `.` in it.
    for (key, tensor) in staged {
        envelope.insert_tensor(key, tensor);
    }
    envelope.set_section(SECTION, text)
}

/// Write a whole optimizer's state into `envelope`, keyed by `model`'s dotted
/// paths — what `Engine::save` is.
///
/// A parameter this optimizer has never updated is simply absent, so a resumed
/// run treats its next update as a first one.
pub(crate) fn store<R: Rule>(
    envelope: &mut Envelope,
    hypers: &[(&'static str, f64)],
    steps: u64,
    model: &dyn Module,
    state: &States<R::Buffers>,
) -> Result<()> {
    let paths = engine::param_paths(model);
    let outgoing: Vec<OutgoingParam<'_>> = paths
        .iter()
        .filter_map(|(path, key)| {
            let entry = state.get(key)?;
            Some(OutgoingParam {
                path,
                clock: entry.clock,
                buffers: R::buffers(&entry.buffers),
            })
        })
        .collect();
    save(envelope, R::KIND, hypers, steps, &outgoing)
}

/// One parameter's state on the way in.
#[derive(Debug)]
pub(crate) struct IncomingParam {
    pub(crate) clock: u64,
    pub(crate) buffers: BTreeMap<String, HostTensor>,
}

/// A parsed, validated optimizer-state section plus its tensors.
#[derive(Debug)]
pub(crate) struct Incoming {
    steps: u64,
    hypers: BTreeMap<String, f64>,
    params: BTreeMap<String, IncomingParam>,
}

impl Incoming {
    /// The number of `step` calls the saved optimizer had made.
    pub(crate) fn steps(&self) -> u64 {
        self.steps
    }

    /// Per-parameter state, keyed by dotted path.
    pub(crate) fn params(&self) -> &BTreeMap<String, IncomingParam> {
        &self.params
    }

    /// A saved hyperparameter, or [`Error::Persistence`] if absent.
    pub(crate) fn hyper(&self, name: &str) -> Result<f64> {
        self.hypers
            .get(name)
            .copied()
            .ok_or_else(|| Error::Persistence {
                msg: format!("optimizer state is missing hyperparameter `{name}`"),
            })
    }

    /// Require the saved hyperparameter set to be exactly `names`: an extra or
    /// missing name means the file was written by a differently-configured
    /// optimizer, which is reported rather than half-applied.
    pub(crate) fn expect_hypers(&self, names: &[&str]) -> Result<()> {
        for name in self.hypers.keys() {
            if !names.contains(&name.as_str()) {
                return Err(Error::Persistence {
                    msg: format!("optimizer state carries unknown hyperparameter `{name}`"),
                });
            }
        }
        for name in names {
            self.hyper(name)?;
        }
        Ok(())
    }
}

/// Read the optimizer state written by [`save`], checking `kind`.
pub(crate) fn load(envelope: &Envelope, kind: &str) -> Result<Incoming> {
    let text = envelope
        .section(SECTION)
        .ok_or_else(|| Error::Persistence {
            msg: format!("envelope carries no `{SECTION}` section"),
        })?;

    let mut version = None;
    let mut found_kind = None;
    let mut steps = None;
    let mut hypers = BTreeMap::new();
    let mut clocks: BTreeMap<String, u64> = BTreeMap::new();
    let mut keys = std::collections::BTreeSet::new();

    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (key, value) = line.split_once('=').ok_or_else(|| Error::Persistence {
            msg: format!("malformed optimizer state line {line:?} (expected key=value)"),
        })?;
        if !keys.insert(key) {
            return Err(Error::Persistence {
                msg: format!("duplicate optimizer state key `{key}`"),
            });
        }
        if key == "version" {
            version = Some(number::<u32>(key, value)?);
        } else if key == "kind" {
            found_kind = Some(value.to_string());
        } else if key == "steps" {
            let value = number::<u64>(key, value)?;
            if value == u64::MAX {
                return Err(Error::Persistence {
                    msg: format!(
                        "optimizer state `steps` is {value}, which cannot be advanced safely"
                    ),
                });
            }
            steps = Some(value);
        } else if let Some(name) = key.strip_prefix("hyper.") {
            hypers.insert(name.to_string(), number::<f64>(key, value)?);
        } else if let Some(path) = key.strip_prefix("clock.") {
            let value = number::<u64>(key, value)?;
            if value == u64::MAX {
                return Err(Error::Persistence {
                    msg: format!(
                        "optimizer state `clock.{path}` is {value}, which cannot be advanced safely"
                    ),
                });
            }
            clocks.insert(path.to_string(), value);
        } else {
            return Err(Error::Persistence {
                msg: format!("unknown optimizer state key `{key}`"),
            });
        }
    }

    match version {
        Some(ENCODING_VERSION) => {}
        Some(other) => {
            return Err(Error::Persistence {
                msg: format!(
                    "optimizer state encoding version {other} (this build reads {ENCODING_VERSION})"
                ),
            });
        }
        None => {
            return Err(Error::Persistence {
                msg: "optimizer state is missing `version`".to_string(),
            });
        }
    }
    match found_kind.as_deref() {
        Some(found) if found == kind => {}
        Some(found) => {
            return Err(Error::Persistence {
                msg: format!("optimizer state was saved by `{found}`, loaded into `{kind}`"),
            });
        }
        None => {
            return Err(Error::Persistence {
                msg: "optimizer state is missing `kind`".to_string(),
            });
        }
    }
    let steps = steps.ok_or_else(|| Error::Persistence {
        msg: "optimizer state is missing `steps`".to_string(),
    })?;

    let mut params: BTreeMap<String, IncomingParam> = clocks
        .into_iter()
        .map(|(path, clock)| {
            (
                path,
                IncomingParam {
                    clock,
                    buffers: BTreeMap::new(),
                },
            )
        })
        .collect();

    for (key, tensor) in envelope.tensors() {
        let Some(rest) = key.strip_prefix(TENSOR_PREFIX) else {
            // A model weight sharing the envelope: not this layer's business.
            continue;
        };
        let (path, name) = rest.rsplit_once('.').ok_or_else(|| Error::Persistence {
            msg: format!("optimizer tensor `{key}` has no `<path>.<buffer>` form"),
        })?;
        let entry = params.get_mut(path).ok_or_else(|| Error::Persistence {
            msg: format!(
                "optimizer tensor `{key}` has no step clock: the `{SECTION}` section \
                 does not mention parameter `{path}`"
            ),
        })?;
        entry.buffers.insert(name.to_string(), tensor.clone());
    }

    Ok(Incoming {
        steps,
        hypers,
        params,
    })
}

/// Rebuild a whole optimizer's per-parameter state from `incoming`, resolving
/// each saved path against `model` — the half of `Engine::load` that walks the
/// file, with [`Rule::decode`] doing each parameter's own buffers.
///
/// Nothing of the optimizer's state is touched here. The caller replaces its
/// state with the returned map only once **every** parameter has decoded, which
/// is what makes a load all-or-nothing.
pub(crate) fn restore<R: Rule>(
    model: &dyn Module,
    incoming: &Incoming,
    rule: &R,
    groups: &Groups<R::Hyper>,
) -> Result<States<R::Buffers>> {
    let values = engine::param_values(model);
    let keys: HashMap<String, GradKey> = engine::param_paths(model).into_iter().collect();
    let mut restored = HashMap::new();
    for (path, saved) in incoming.params() {
        let (key, value) = locate(&keys, &values, path)?;
        restored.insert(
            key,
            ParamState {
                clock: saved.clock,
                buffers: rule.decode(Decode {
                    path,
                    saved,
                    value,
                    incoming,
                    groups,
                })?,
            },
        );
    }
    Ok(restored)
}

/// Resolve a saved path against the target model, yielding the state key to
/// file it under and the parameter value to validate buffers against.
///
/// A path the model does not have means the checkpoint belongs to a different
/// model — reported rather than ignored, because silently dropping a moment
/// buffer changes the trajectory of a resumed run.
fn locate<'a>(
    keys: &HashMap<String, GradKey>,
    values: &'a HashMap<String, Tensor>,
    path: &str,
) -> Result<(GradKey, &'a Tensor)> {
    match (keys.get(path), values.get(path)) {
        (Some(key), Some(value)) => Ok((*key, value)),
        _ => Err(Error::Persistence {
            msg: format!(
                "optimizer state names parameter `{path}`, which this model does not have"
            ),
        }),
    }
}

/// The rejection for a buffer name an optimizer does not know.
pub(crate) fn unknown_buffer(kind: &str, path: &str, name: &str) -> Error {
    Error::Persistence {
        msg: format!("optimizer state has unknown buffer `{name}` for `{path}` (kind {kind})"),
    }
}

/// Rebuild one loaded moment buffer as a tensor beside the parameter it
/// belongs to, checking it against that parameter.
///
/// A moment buffer must have the parameter's dims and its accumulation dtype
/// (see `DType::accumulation_dtype`) — a mismatch means the checkpoint belongs to a
/// different model and is rejected before any of the optimizer's state is
/// replaced.
pub(crate) fn restore_buffer(
    host: &HostTensor,
    param: &Tensor,
    path: &str,
    name: &str,
) -> Result<Tensor> {
    let expected = param.dtype().accumulation_dtype();
    if host.dtype() != expected {
        return Err(Error::Persistence {
            msg: format!(
                "optimizer buffer `{path}.{name}` has dtype {}, expected {expected}",
                host.dtype()
            ),
        });
    }
    if host.dims() != param.dims() {
        return Err(Error::Persistence {
            msg: format!(
                "optimizer buffer `{path}.{name}` has shape {:?}, expected {:?}",
                host.dims(),
                param.dims()
            ),
        });
    }
    from_host_tensor(host, &param.device())
}

/// Parse one numeric field, naming the key on failure.
fn number<T: std::str::FromStr>(key: &str, value: &str) -> Result<T> {
    value.parse::<T>().map_err(|_| Error::Persistence {
        msg: format!("optimizer state `{key}` is not a number: {value:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::optim::testkit::{CPU, t};

    /// An envelope whose `optimizer` section is exactly `text`.
    fn with_section(text: &str) -> Envelope {
        let mut e = Envelope::new();
        e.set_section(SECTION, text).unwrap();
        e
    }

    fn saved() -> Envelope {
        let velocity = t(&[0.5, -0.25]);
        let mut envelope = Envelope::new();
        save(
            &mut envelope,
            "sgd",
            &[("lr", 3e-4), ("momentum", 0.9)],
            7,
            &[
                OutgoingParam {
                    path: "fc1.weight",
                    clock: 7,
                    buffers: vec![("velocity", &velocity)],
                },
                OutgoingParam {
                    path: "fc1.bias",
                    clock: 2,
                    buffers: Vec::new(),
                },
            ],
        )
        .unwrap();
        envelope
    }

    #[test]
    fn the_section_round_trips_verbatim() {
        let envelope = saved();
        let incoming = load(&envelope, "sgd").unwrap();
        assert_eq!(incoming.steps(), 7);
        // `{:?}` is the shortest representation that parses back exactly, so a
        // resumed run gets bit-identical hyperparameters.
        assert_eq!(incoming.hyper("lr").unwrap(), 3e-4);
        assert_eq!(incoming.hyper("momentum").unwrap(), 0.9);
        incoming.expect_hypers(&["lr", "momentum"]).unwrap();

        let params = incoming.params();
        assert_eq!(params.len(), 2);
        assert_eq!(params["fc1.weight"].clock, 7);
        assert_eq!(params["fc1.bias"].clock, 2);
        // A parameter with no buffer keeps its clock and nothing else.
        assert!(params["fc1.bias"].buffers.is_empty());
        let buffer = &params["fc1.weight"].buffers["velocity"];
        assert_eq!(buffer.dtype(), DType::F32);
        assert_eq!(buffer.dims(), &[2]);
        // Tensors live under the reserved namespace, so a model's own
        // state-dict keys can share one envelope.
        assert_eq!(
            envelope.tensors().keys().collect::<Vec<_>>(),
            vec!["optim.fc1.weight.velocity"]
        );
    }

    #[test]
    fn a_buffer_restores_beside_its_parameter() {
        let envelope = saved();
        let incoming = load(&envelope, "sgd").unwrap();
        let host = &incoming.params()["fc1.weight"].buffers["velocity"];
        let param = t(&[0.0, 0.0]);
        let restored = restore_buffer(host, &param, "fc1.weight", "velocity").unwrap();
        assert_eq!(restored.to_vec::<f32>().unwrap(), vec![0.5, -0.25]);
        assert_eq!(restored.device(), CPU);

        // A parameter of a different shape means a different model.
        let msg = restore_buffer(host, &t(&[0.0]), "fc1.weight", "velocity")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("`fc1.weight.velocity` has shape"), "{msg}");

        // …and so does a different accumulation dtype.
        let f64_param = Tensor::zeros([2], DType::F64, &CPU).unwrap();
        let msg = restore_buffer(host, &f64_param, "fc1.weight", "velocity")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("dtype f32, expected f64"), "{msg}");
    }

    #[test]
    fn the_wrong_optimizer_is_rejected() {
        let msg = load(&saved(), "adam").unwrap_err().to_string();
        assert!(msg.contains("saved by `sgd`, loaded into `adam`"), "{msg}");
    }

    #[test]
    fn a_missing_section_is_rejected() {
        let msg = load(&Envelope::new(), "sgd").unwrap_err().to_string();
        assert!(msg.contains("no `optimizer` section"), "{msg}");
    }

    #[test]
    fn every_malformed_section_is_named() {
        let cases = [
            (
                "no equals sign",
                "version=1\nkind=sgd\nsteps=1\nrubbish\n",
                "malformed",
            ),
            (
                "unknown key",
                "version=1\nkind=sgd\nsteps=1\nmystery=3\n",
                "unknown optimizer state key `mystery`",
            ),
            (
                "stale version",
                "version=99\nkind=sgd\nsteps=1\n",
                "encoding version 99",
            ),
            ("no version", "kind=sgd\nsteps=1\n", "missing `version`"),
            ("no kind", "version=1\nsteps=1\n", "missing `kind`"),
            ("no steps", "version=1\nkind=sgd\n", "missing `steps`"),
            (
                "non-numeric steps",
                "version=1\nkind=sgd\nsteps=soon\n",
                "`steps` is not a number",
            ),
            (
                "non-numeric hyper",
                "version=1\nkind=sgd\nsteps=1\nhyper.lr=fast\n",
                "`hyper.lr` is not a number",
            ),
            (
                "non-numeric clock",
                "version=1\nkind=sgd\nsteps=1\nclock.w=later\n",
                "`clock.w` is not a number",
            ),
        ];
        for (label, text, expected) in cases {
            let msg = load(&with_section(text), "sgd").unwrap_err().to_string();
            assert!(msg.contains(expected), "{label}: {msg}");
        }
    }

    #[test]
    fn exhausted_loaded_clocks_are_rejected_by_field_name() {
        let max = u64::MAX;
        for (label, text, expected) in [
            (
                "global clock",
                format!("version=1\nkind=sgd\nsteps={max}\n"),
                "`steps`",
            ),
            (
                "parameter clock",
                format!("version=1\nkind=sgd\nsteps=1\nclock.fc.weight={max}\n"),
                "`clock.fc.weight`",
            ),
        ] {
            let err = load(&with_section(&text), "sgd").unwrap_err();
            assert!(matches!(err, Error::Persistence { .. }), "{label}: {err}");
            let msg = err.to_string();
            assert!(msg.contains(expected), "{label}: {msg}");
            assert!(msg.contains("cannot be advanced safely"), "{label}: {msg}");
        }
    }

    #[test]
    fn duplicate_optimizer_fields_are_rejected_instead_of_last_wins() {
        let cases = [
            "version=1\nversion=1\nkind=sgd\nsteps=1\n",
            "version=1\nkind=sgd\nkind=adam\nsteps=1\n",
            "version=1\nkind=sgd\nsteps=1\nsteps=2\n",
            "version=1\nkind=sgd\nsteps=1\nhyper.lr=0.1\nhyper.lr=0.2\n",
            "version=1\nkind=sgd\nsteps=1\nclock.w=1\nclock.w=2\n",
        ];
        for text in cases {
            let message = load(&with_section(text), "sgd").unwrap_err().to_string();
            assert!(
                message.contains("duplicate optimizer state key"),
                "{message}"
            );
        }
    }

    #[test]
    fn near_max_loaded_clocks_preserve_their_exact_values() {
        let near_max = u64::MAX - 1;
        let text = format!(
            "version=1\nkind=sgd\nsteps={near_max}\nhyper.lr=0.1\nclock.fc.weight={near_max}\n"
        );
        let incoming = load(&with_section(&text), "sgd").unwrap();
        assert_eq!(incoming.steps(), near_max);
        assert_eq!(incoming.params()["fc.weight"].clock, near_max);
    }

    #[test]
    fn a_buffer_with_no_step_clock_is_rejected() {
        let mut envelope = with_section("version=1\nkind=sgd\nsteps=1\n");
        envelope.insert_tensor(
            "optim.fc1.weight.velocity",
            HostTensor::from_bytes(DType::F32, vec![1], vec![0u8; 4]).unwrap(),
        );
        let msg = load(&envelope, "sgd").unwrap_err().to_string();
        assert!(msg.contains("has no step clock"), "{msg}");
    }

    #[test]
    fn a_buffer_key_without_a_buffer_name_is_rejected() {
        let mut envelope = with_section("version=1\nkind=sgd\nsteps=1\nclock.w=1\n");
        envelope.insert_tensor(
            "optim.solo",
            HostTensor::from_bytes(DType::F32, vec![1], vec![0u8; 4]).unwrap(),
        );
        let msg = load(&envelope, "sgd").unwrap_err().to_string();
        assert!(msg.contains("has no `<path>.<buffer>` form"), "{msg}");
    }

    #[test]
    fn a_model_tensor_sharing_the_envelope_is_ignored() {
        let mut envelope = saved();
        envelope.insert_tensor(
            "fc1.weight",
            HostTensor::from_bytes(DType::F32, vec![2], vec![0u8; 8]).unwrap(),
        );
        // The model's own weight is not optimizer state; the load must not trip
        // over it, which is what makes one-file checkpoints possible.
        let incoming = load(&envelope, "sgd").unwrap();
        assert_eq!(incoming.params().len(), 2);
    }

    #[test]
    fn expect_hypers_is_exact() {
        let incoming = load(&saved(), "sgd").unwrap();
        // An extra name the file does not carry.
        let msg = incoming
            .expect_hypers(&["lr", "momentum", "nesterov"])
            .unwrap_err()
            .to_string();
        assert!(msg.contains("missing hyperparameter `nesterov`"), "{msg}");
        // A name the file carries that this optimizer does not know.
        let msg = incoming.expect_hypers(&["lr"]).unwrap_err().to_string();
        assert!(msg.contains("unknown hyperparameter `momentum`"), "{msg}");
    }

    #[test]
    fn a_path_the_encoding_cannot_represent_is_refused() {
        let velocity = t(&[0.0]);
        for path in ["odd=name", "odd\nname"] {
            let mut envelope = Envelope::new();
            let err = save(
                &mut envelope,
                "sgd",
                &[],
                1,
                &[OutgoingParam {
                    path,
                    clock: 1,
                    buffers: vec![("velocity", &velocity)],
                }],
            )
            .unwrap_err();
            assert!(err.to_string().contains("cannot represent"), "{err}");
        }
    }

    #[test]
    fn a_refused_save_leaves_the_envelope_untouched() {
        // The good parameter is written first, so a one-pass implementation
        // would leave its buffer behind when the second is refused — and the
        // next attempt on that envelope would then fail with the misleading
        // "already carries optimizer state" instead of succeeding.
        let velocity = t(&[1.0]);
        let mut envelope = Envelope::new();
        assert!(
            save(
                &mut envelope,
                "sgd",
                &[("lr", 0.1)],
                1,
                &[
                    OutgoingParam {
                        path: "fine",
                        clock: 1,
                        buffers: vec![("velocity", &velocity)],
                    },
                    OutgoingParam {
                        path: "odd=name",
                        clock: 1,
                        buffers: vec![("velocity", &velocity)],
                    },
                ],
            )
            .is_err()
        );
        assert!(envelope.tensors().is_empty(), "{:?}", envelope.tensors());
        assert!(envelope.section(SECTION).is_none());

        // …so retrying with the problem fixed just works.
        save(
            &mut envelope,
            "sgd",
            &[("lr", 0.1)],
            1,
            &[OutgoingParam {
                path: "fine",
                clock: 1,
                buffers: vec![("velocity", &velocity)],
            }],
        )
        .unwrap();
        assert_eq!(load(&envelope, "sgd").unwrap().params().len(), 1);
    }

    #[test]
    fn duplicate_outgoing_fields_and_buffers_are_refused() {
        let velocity = t(&[1.0]);
        let mut envelope = Envelope::new();
        let duplicate_hyper =
            save(&mut envelope, "sgd", &[("lr", 0.1), ("lr", 0.2)], 1, &[]).unwrap_err();
        assert!(
            duplicate_hyper
                .to_string()
                .contains("duplicate optimizer state key")
        );

        let duplicate_buffer = save(
            &mut envelope,
            "sgd",
            &[],
            1,
            &[OutgoingParam {
                path: "w",
                clock: 1,
                buffers: vec![("velocity", &velocity), ("velocity", &velocity)],
            }],
        )
        .unwrap_err();
        assert!(
            duplicate_buffer
                .to_string()
                .contains("duplicate optimizer buffer")
        );
        assert!(envelope.tensors().is_empty());
        assert!(envelope.section(SECTION).is_none());
    }

    #[test]
    fn a_second_save_into_one_envelope_is_refused() {
        let mut envelope = saved();
        let err = save(&mut envelope, "sgd", &[], 1, &[]).unwrap_err();
        assert!(
            err.to_string().contains("already carries optimizer state"),
            "{err}"
        );

        // Also when only the tensors are there (a truncated earlier write).
        let mut envelope = Envelope::new();
        envelope.insert_tensor(
            "optim.w.velocity",
            HostTensor::from_bytes(DType::F32, vec![1], vec![0u8; 4]).unwrap(),
        );
        assert!(save(&mut envelope, "sgd", &[], 1, &[]).is_err());
    }

    #[test]
    fn an_unknown_buffer_name_names_the_parameter() {
        let msg = unknown_buffer("sgd", "fc1.weight", "second_moment").to_string();
        assert!(msg.contains("`second_moment`"), "{msg}");
        assert!(msg.contains("`fc1.weight`"), "{msg}");
    }

    #[test]
    fn locate_rejects_a_path_the_model_does_not_have() {
        let keys = std::collections::HashMap::new();
        let values = std::collections::HashMap::new();
        let msg = locate(&keys, &values, "ghost").unwrap_err().to_string();
        assert!(
            msg.contains("`ghost`, which this model does not have"),
            "{msg}"
        );
    }
}
