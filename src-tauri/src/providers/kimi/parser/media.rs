//! Media references in kimi-code wires.
//!
//! Besides plain paths and `data:` URLs, a media part's `url` can name its
//! bytes indirectly:
//!
//! * `blobref:<mime>;<sha256>` — inline base64 of 4096+ characters (an image
//!   sent over ACP, a tool's image output) that kimi-code offloaded from
//!   `context.append_message` / `context.append_loop_event` records into the
//!   agent's blob store, `<agent dir>/blobs/<sha256>`.
//! * `kimi-file://<file id>` — an attachment uploaded from the TUI (a pasted
//!   image). The session keeps its copy in `media/`, named by the `key` in
//!   `media/meta/<file id>.json`.
//!
//! Resolving both to local paths lets the shared renderer emit
//! `[Image: source: <path>]` markers the UI can load from disk.

use std::path::Path;

use serde_json::Value;

const BLOBREF_PREFIX: &str = "blobref:";
const KIMI_FILE_PREFIX: &str = "kimi-file://";

/// Record types whose content parts can carry media references.
pub(super) fn may_carry_media_refs(record_type: &str) -> bool {
    matches!(
        record_type,
        "context.append_message" | "context.append_loop_event"
    )
}

/// Rewrite every media `url` holding a reference to the local file it names.
/// Returns how many references name no existing file; those stay as written.
pub(super) fn resolve_media_refs(value: &mut Value, agent_dir: &Path) -> u32 {
    match value {
        Value::Object(map) => {
            let mut unresolved = 0;
            if let Some(Value::String(url)) = map.get_mut("url")
                && (url.starts_with(BLOBREF_PREFIX) || url.starts_with(KIMI_FILE_PREFIX))
            {
                match local_path(url, agent_dir) {
                    Some(path) => *url = path,
                    None => {
                        log::warn!("Kimi media reference '{url}' has no local file");
                        unresolved += 1;
                    }
                }
            }
            for child in map.values_mut() {
                unresolved += resolve_media_refs(child, agent_dir);
            }
            unresolved
        }
        Value::Array(items) => items
            .iter_mut()
            .map(|item| resolve_media_refs(item, agent_dir))
            .sum(),
        _ => 0,
    }
}

fn local_path(url: &str, agent_dir: &Path) -> Option<String> {
    let path = if let Some(reference) = url.strip_prefix(BLOBREF_PREFIX) {
        // `<mime>;<sha256>`: only a plain hex digest names a blob.
        let (_mime, hash) = reference.split_once(';')?;
        if hash.is_empty() || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return None;
        }
        agent_dir.join("blobs").join(hash)
    } else {
        let file_id = url.strip_prefix(KIMI_FILE_PREFIX)?;
        if !is_plain_name(file_id) {
            return None;
        }
        // `<session>/agents/<agent>/wire.jsonl` → `<session>/media/`.
        let media = agent_dir.parent()?.parent()?.join("media");
        let meta =
            std::fs::read_to_string(media.join("meta").join(format!("{file_id}.json"))).ok()?;
        let meta: Value = serde_json::from_str(&meta).ok()?;
        let key = meta.get("key")?.as_str()?;
        if !is_plain_name(key) {
            return None;
        }
        media.join(key)
    };
    path.is_file().then(|| path.to_string_lossy().into_owned())
}

/// A single file-name component: never a path, so a reference cannot reach
/// outside the directory it is resolved in.
fn is_plain_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\'])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn references_resolve_to_local_files() {
        let root = tempfile::tempdir().unwrap();
        let session_dir = root.path().join("session_x");
        let agent_dir = session_dir.join("agents").join("main");
        let media = session_dir.join("media");
        std::fs::create_dir_all(agent_dir.join("blobs")).unwrap();
        std::fs::create_dir_all(media.join("meta")).unwrap();
        std::fs::write(agent_dir.join("blobs").join("ab12"), b"png").unwrap();
        std::fs::write(media.join("f_1.png"), b"png").unwrap();
        std::fs::write(
            media.join("meta").join("f_1.json"),
            r#"{"version":1,"key":"f_1.png","name":"pasted-image.png","mediaType":"image/png"}"#,
        )
        .unwrap();
        std::fs::write(
            media.join("meta").join("f_evil.json"),
            r#"{"version":1,"key":"../../secret.png","name":"x.png","mediaType":"image/png"}"#,
        )
        .unwrap();
        let mut record = json!({
            "type": "context.append_message",
            "message": {"content": [
                {"type": "image_url", "imageUrl": {"url": "blobref:image/png;ab12"}},
                {"type": "image_url", "imageUrl": {"url": "kimi-file://f_1"}},
                {"type": "image_url", "imageUrl": {"url": "blobref:image/png;ffff"}},
                {"type": "image_url", "imageUrl": {"url": "blobref:image/png;../../etc/passwd"}},
                {"type": "image_url", "imageUrl": {"url": "kimi-file://f_gone"}},
                {"type": "image_url", "imageUrl": {"url": "kimi-file://f_evil"}},
                {"type": "image_url", "imageUrl": {"url": "kimi-file://../meta/f_1"}},
                {"type": "image_url", "imageUrl": {"url": "https://example.com/a.png"}}
            ]}
        });

        assert_eq!(resolve_media_refs(&mut record, &agent_dir), 5);
        let urls: Vec<&str> = record["message"]["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|part| part["imageUrl"]["url"].as_str().unwrap())
            .collect();
        let blob = agent_dir.join("blobs").join("ab12");
        let pasted = media.join("f_1.png");
        assert_eq!(
            urls,
            [
                blob.to_string_lossy().as_ref(),
                pasted.to_string_lossy().as_ref(),
                "blobref:image/png;ffff",
                "blobref:image/png;../../etc/passwd",
                "kimi-file://f_gone",
                "kimi-file://f_evil",
                "kimi-file://../meta/f_1",
                "https://example.com/a.png",
            ]
        );
    }
}
