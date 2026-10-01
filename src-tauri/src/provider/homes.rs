//! Extra home directories.
//!
//! Providers resolve their data from the user's home (`~/.claude`,
//! `~/.local/share/opencode`, …) plus per-tool env overrides. A WSL process
//! can also reach the Windows side's trees under `/mnt/c/Users/<name>`, so
//! `SESSIONVIEW_EXTRA_HOMES` (a PATH-style list) names further homes to index
//! with the same binary.
//!
//! [`runtime`] builds each provider from its primary instance — exactly what
//! a single-home process uses, env overrides included — plus one instance per
//! extra home at the tool's default layout there. Without extra homes the
//! primary instance is the runtime itself; otherwise [`MultiHome`] fans the
//! trait out over all of them.

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::models::Provider;

use super::{
    LoadedSession, ParsedSession, ProviderError, ScanOutcome, SessionProvider, SourceState,
};

const EXTRA_HOMES_ENV: &str = "SESSIONVIEW_EXTRA_HOMES";

/// Homes listed in `SESSIONVIEW_EXTRA_HOMES`, read once per process.
pub(crate) fn extra_homes() -> &'static [PathBuf] {
    static EXTRA_HOMES: OnceLock<Vec<PathBuf>> = OnceLock::new();
    EXTRA_HOMES.get_or_init(|| match std::env::var_os(EXTRA_HOMES_ENV) {
        Some(value) => parse_extra_homes(&value, dirs::home_dir().as_deref()),
        None => Vec::new(),
    })
}

/// The real home followed by every extra home: each directory whose tool
/// trees this process indexes.
pub(crate) fn all_homes() -> Vec<PathBuf> {
    dirs::home_dir()
        .into_iter()
        .chain(extra_homes().iter().cloned())
        .collect()
}

/// Split a PATH-style list, dropping empty entries, repeats, and the real
/// home (already the primary). Relative entries would resolve against the
/// working directory and a filesystem root would allowlist the whole disk
/// for file reads, so both are skipped with a warning. A home that does not
/// exist yet is kept (a mount may appear later) but reported.
fn parse_extra_homes(value: &OsStr, home: Option<&Path>) -> Vec<PathBuf> {
    let mut homes: Vec<PathBuf> = Vec::new();
    for entry in std::env::split_paths(value) {
        if entry.as_os_str().is_empty() || home == Some(entry.as_path()) || homes.contains(&entry) {
            continue;
        }
        if !entry.is_absolute() || entry.parent().is_none() {
            log::warn!(
                "ignoring {EXTRA_HOMES_ENV} entry '{}': a home must be an absolute path below the filesystem root",
                entry.display()
            );
            continue;
        }
        if !entry.is_dir() {
            log::warn!(
                "{EXTRA_HOMES_ENV} entry '{}' is not a directory yet",
                entry.display()
            );
        }
        homes.push(entry);
    }
    homes
}

/// Build one provider's runtime from its primary instance and `for_home`,
/// which yields the instances reading another home's default tool layout.
pub(crate) fn runtime<P, I>(
    primary: Option<P>,
    for_home: impl Fn(&Path) -> I,
) -> Option<Box<dyn SessionProvider>>
where
    P: SessionProvider + 'static,
    I: IntoIterator<Item = P>,
{
    runtime_with_homes(extra_homes(), primary, for_home)
}

fn runtime_with_homes<P, I>(
    extra_homes: &[PathBuf],
    primary: Option<P>,
    for_home: impl Fn(&Path) -> I,
) -> Option<Box<dyn SessionProvider>>
where
    P: SessionProvider + 'static,
    I: IntoIterator<Item = P>,
{
    let mut instances: Vec<Box<dyn SessionProvider>> = primary
        .into_iter()
        .chain(extra_homes.iter().flat_map(|home| for_home(home)))
        .map(|provider| Box::new(provider) as Box<dyn SessionProvider>)
        .collect();
    if instances.len() <= 1 {
        return instances.pop();
    }
    Some(Box::new(MultiHome { instances }))
}

/// One provider fanned out over its per-home instances, primary first.
///
/// - Scans concatenate in instance order. The index holds one row per
///   session id, so an id surfaced twice in one pass (overlapping trees, a
///   history copied between homes) keeps its first occurrence and the rest
///   are skipped with a warning. Copies only collide when they parse in the
///   same pass: an incremental pass that re-parses just the later copy still
///   upserts it.
/// - A scan error from the primary instance fails the scan as it would in a
///   single-home process. An extra home's error is logged and that home is
///   skipped for the pass, so an unreadable mount cannot stall the rest; its
///   indexed sessions stay while their files remain on disk.
/// - Loads route to the instance owning the longest source root that
///   contains the session's source path; a path no root claims goes to the
///   primary instance, as in a single-home process.
struct MultiHome {
    instances: Vec<Box<dyn SessionProvider>>,
}

impl MultiHome {
    fn route(&self, source_path: &str) -> &dyn SessionProvider {
        let path = Path::new(source_path);
        let mut best: Option<(usize, &dyn SessionProvider)> = None;
        for instance in &self.instances {
            for root in instance.source_roots() {
                let depth = root.components().count();
                if path.starts_with(&root) && best.is_none_or(|(best_depth, _)| depth > best_depth)
                {
                    best = Some((depth, instance.as_ref()));
                }
            }
        }
        best.map_or(self.instances[0].as_ref(), |(_, instance)| instance)
    }

    /// Run `scan` on every instance, isolating extra homes' errors.
    fn scan_each<T>(
        &self,
        scan: impl Fn(&dyn SessionProvider) -> Result<T, ProviderError>,
    ) -> Result<Vec<T>, ProviderError> {
        let mut outcomes = Vec::new();
        for (index, instance) in self.instances.iter().enumerate() {
            match scan(instance.as_ref()) {
                Ok(outcome) => outcomes.push(outcome),
                Err(error) if index > 0 => log::warn!(
                    "skipping an extra home for {} this pass ({:?}): {error}",
                    self.provider().key(),
                    instance.source_roots()
                ),
                Err(error) => return Err(error),
            }
        }
        Ok(outcomes)
    }

    fn keep_first_per_id(&self, sessions: Vec<ParsedSession>) -> Vec<ParsedSession> {
        let mut seen: HashMap<String, String> = HashMap::new();
        sessions
            .into_iter()
            .filter(|session| match seen.get(&session.meta.id) {
                Some(kept) => {
                    log::warn!(
                        "{} session '{}' found in both '{kept}' and '{}'; indexing the first",
                        self.provider().key(),
                        session.meta.id,
                        session.meta.source_path
                    );
                    false
                }
                None => {
                    seen.insert(session.meta.id.clone(), session.meta.source_path.clone());
                    true
                }
            })
            .collect()
    }
}

impl SessionProvider for MultiHome {
    fn provider(&self) -> Provider {
        self.instances[0].provider()
    }

    fn source_roots(&self) -> Vec<PathBuf> {
        let mut roots: Vec<PathBuf> = Vec::new();
        for root in self
            .instances
            .iter()
            .flat_map(|instance| instance.source_roots())
        {
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
        roots
    }

    fn scan_all(&self) -> Result<Vec<ParsedSession>, ProviderError> {
        let parsed = self.scan_each(|instance| instance.scan_all())?;
        Ok(self.keep_first_per_id(parsed.into_iter().flatten().collect()))
    }

    fn scan_incremental(
        &self,
        known: &HashMap<String, SourceState>,
    ) -> Result<ScanOutcome, ProviderError> {
        let mut parsed = Vec::new();
        let mut unchanged_source_paths = Vec::new();
        let mut seen_unchanged = HashSet::new();
        for outcome in self.scan_each(|instance| instance.scan_incremental(known))? {
            parsed.extend(outcome.parsed);
            for path in outcome.unchanged_source_paths {
                if seen_unchanged.insert(path.clone()) {
                    unchanged_source_paths.push(path);
                }
            }
        }
        Ok(ScanOutcome {
            parsed: self.keep_first_per_id(parsed),
            unchanged_source_paths,
        })
    }

    fn load_messages(
        &self,
        session_id: &str,
        source_path: &str,
    ) -> Result<LoadedSession, ProviderError> {
        self.route(source_path)
            .load_messages(session_id, source_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Message;
    use crate::providers::dsh::DshProvider;

    #[test]
    fn parse_extra_homes_skips_empty_repeated_relative_root_and_real_home() {
        let base = std::env::temp_dir();
        let (real, a, b) = (base.join("real"), base.join("a"), base.join("b"));
        let root = base.ancestors().last().unwrap().to_path_buf();
        let joined = std::env::join_paths([
            a.clone(),
            PathBuf::new(),
            real.clone(),
            PathBuf::from("relative").join("home"),
            root,
            b.clone(),
            a.clone(),
        ])
        .unwrap();
        assert_eq!(parse_extra_homes(&joined, Some(&real)), vec![a, b]);
    }

    /// Records which instance served a load.
    struct Named {
        name: &'static str,
        root: PathBuf,
    }

    impl SessionProvider for Named {
        fn provider(&self) -> Provider {
            Provider::Claude
        }
        fn source_roots(&self) -> Vec<PathBuf> {
            vec![self.root.clone()]
        }
        fn scan_all(&self) -> Result<Vec<ParsedSession>, ProviderError> {
            Ok(Vec::new())
        }
        fn load_messages(&self, _: &str, _: &str) -> Result<LoadedSession, ProviderError> {
            Ok(LoadedSession::new(vec![Message::user(self.name)]))
        }
    }

    fn served_by(runtime: &dyn SessionProvider, source_path: &Path) -> String {
        runtime
            .load_messages("id", source_path.to_string_lossy().as_ref())
            .unwrap()
            .messages[0]
            .content
            .clone()
    }

    #[test]
    fn without_extra_homes_the_primary_is_the_runtime() {
        let base = std::env::temp_dir();
        let runtime = runtime_with_homes(
            &[],
            Some(Named {
                name: "primary",
                root: base.join("home").join(".claude"),
            }),
            |_| -> [Named; 0] { unreachable!("no extra homes") },
        )
        .unwrap();
        assert_eq!(
            served_by(runtime.as_ref(), &base.join("x.jsonl")),
            "primary"
        );
        assert!(runtime_with_homes::<Named, [Named; 0]>(&[], None, |_| []).is_none());
    }

    #[test]
    fn loads_route_to_the_longest_owning_root_else_the_primary() {
        let base = std::env::temp_dir();
        let (home, extra) = (base.join("home"), base.join("extra"));
        let runtime = runtime_with_homes(
            std::slice::from_ref(&extra),
            Some(Named {
                name: "primary",
                root: home.join(".claude"),
            }),
            |home| {
                [
                    Named {
                        name: "extra",
                        root: home.join(".claude"),
                    },
                    Named {
                        name: "extra-projects",
                        root: home.join(".claude").join("projects"),
                    },
                ]
            },
        )
        .unwrap();
        let served = |path: PathBuf| served_by(runtime.as_ref(), &path);
        assert_eq!(served(home.join(".claude").join("x.jsonl")), "primary");
        assert_eq!(served(extra.join(".claude").join("todo.json")), "extra");
        assert_eq!(
            served(extra.join(".claude").join("projects").join("x.jsonl")),
            "extra-projects"
        );
        // A sibling whose name merely extends a root's last component.
        assert_eq!(served(extra.join(".claude-old").join("x.jsonl")), "primary");
        assert_eq!(served(base.join("gone").join("x.jsonl")), "primary");
    }

    /// A home whose scan always fails (an unreadable mount).
    struct Unreadable;

    impl SessionProvider for Unreadable {
        fn provider(&self) -> Provider {
            Provider::Dsh
        }
        fn source_roots(&self) -> Vec<PathBuf> {
            Vec::new()
        }
        fn scan_all(&self) -> Result<Vec<ParsedSession>, ProviderError> {
            Err(ProviderError::Parse("permission denied".into()))
        }
        fn load_messages(&self, _: &str, _: &str) -> Result<LoadedSession, ProviderError> {
            Err(ProviderError::Parse("permission denied".into()))
        }
    }

    #[test]
    fn an_unreadable_extra_home_does_not_stall_the_primary() {
        let primary = tempfile::tempdir().unwrap();
        write_dsh_session(primary.path(), "s-primary", "from primary");
        let runtime = MultiHome {
            instances: vec![
                Box::new(DshProvider::for_home(primary.path())),
                Box::new(Unreadable),
            ],
        };
        assert_eq!(runtime.scan_all().unwrap().len(), 1);
        assert_eq!(
            runtime
                .scan_incremental(&HashMap::new())
                .unwrap()
                .parsed
                .len(),
            1
        );

        let failing_primary = MultiHome {
            instances: vec![
                Box::new(Unreadable),
                Box::new(DshProvider::for_home(primary.path())),
            ],
        };
        assert!(failing_primary.scan_all().is_err());
    }

    fn write_dsh_session(home: &Path, id: &str, prompt: &str) -> PathBuf {
        let dir = home
            .join(".dsh")
            .join("sessions")
            .join("--tmp-p--")
            .join(id);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.jsonl");
        std::fs::write(
            &path,
            format!(
                r#"{{"type":"session","version":0,"id":"{id}","createdAt":1786865077879,"cwd":"/tmp/proj"}}
{{"type":"user/message","seq":1,"time":1786865078000,"data":{{"content":[{{"type":"text","text":"{prompt}"}}],"source":{{"kind":"user"}}}}}}
"#
            ),
        )
        .unwrap();
        path
    }

    fn dsh_runtime(primary: &Path, extra: &Path) -> Box<dyn SessionProvider> {
        runtime_with_homes(
            &[extra.to_path_buf()],
            Some(DshProvider::for_home(primary)),
            |home| [DshProvider::for_home(home)],
        )
        .unwrap()
    }

    #[test]
    fn scans_cover_every_home_and_loads_route_back() {
        let primary = tempfile::tempdir().unwrap();
        let extra = tempfile::tempdir().unwrap();
        let primary_log = write_dsh_session(primary.path(), "s-primary", "from primary");
        let extra_log = write_dsh_session(extra.path(), "s-extra", "from extra");
        let runtime = dsh_runtime(primary.path(), extra.path());

        assert_eq!(runtime.provider(), Provider::Dsh);
        let parsed = runtime.scan_all().unwrap();
        let ids: Vec<&str> = parsed.iter().map(|s| s.meta.id.as_str()).collect();
        assert_eq!(ids, ["s-primary", "s-extra"]);

        let known: HashMap<String, SourceState> = parsed
            .iter()
            .map(|session| {
                (
                    session.meta.source_path.clone(),
                    SourceState {
                        size: session.meta.file_size_bytes,
                        mtime: session.source_mtime,
                        title: Some(session.meta.title.clone()),
                    },
                )
            })
            .collect();
        let outcome = runtime.scan_incremental(&known).unwrap();
        assert!(outcome.parsed.is_empty());
        assert_eq!(outcome.unchanged_source_paths.len(), 2);

        for (id, log, prompt) in [
            ("s-primary", &primary_log, "from primary"),
            ("s-extra", &extra_log, "from extra"),
        ] {
            let loaded = runtime
                .load_messages(id, log.to_string_lossy().as_ref())
                .unwrap();
            assert_eq!(loaded.messages[0].content, prompt);
        }
        assert_eq!(runtime.source_roots().len(), 2);
    }

    #[test]
    fn a_session_id_in_two_homes_indexes_the_primary_copy_once() {
        let primary = tempfile::tempdir().unwrap();
        let extra = tempfile::tempdir().unwrap();
        let primary_log = write_dsh_session(primary.path(), "s-same", "primary copy");
        write_dsh_session(extra.path(), "s-same", "extra copy");
        let runtime = dsh_runtime(primary.path(), extra.path());

        let full = runtime.scan_all().unwrap();
        let incremental = runtime.scan_incremental(&HashMap::new()).unwrap().parsed;
        for parsed in [full, incremental] {
            assert_eq!(parsed.len(), 1);
            assert_eq!(
                parsed[0].meta.source_path,
                primary_log.to_string_lossy().as_ref()
            );
        }
    }
}
