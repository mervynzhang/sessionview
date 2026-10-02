//! DSH (DeepSeek Harness) session-log parser.
//!
//! On-disk format (authoritative spec: `@deepseek-ai/dsh-session` and
//! `@deepseek-ai/dsh-session-persistence-jsonl`):
//!
//! - One session lives in its own directory:
//!   `$DSH_HOME/sessions/<project-key>/<session-id>/session.jsonl[.zstd]`
//!   (`DSH_HOME` defaults to `~/.dsh`). The artifact is zstd-compressed JSONL
//!   when the suffix is `.jsonl.zstd`; uncompressed `.jsonl` is also accepted.
//!   Newer releases persist versioned artifacts instead
//!   (`session.v2.jsonl.zstd`, `session.v3.jsonl.zstd`, …); discovery picks
//!   the highest generation per directory (see `artifact_rank`).
//! - The first record is the immutable session header
//!   `{type:"session", version, id, createdAt, cwd?, parentSession?, origin?, agentPreset?, ...}`.
//!   Headers at version 0–4 share the framing parsed here (`MAX_SUPPORTED_HEADER_VERSION`);
//!   v2+ sessions additionally carry `isSeeded`/`delegationDepth` and a per-record
//!   `surfaceOp: "append"` marker on surface events (the default behavior, ignored).
//! - A seeded (forked) session starts with a copy of its parent's history: the
//!   first `seedLength` events (v0/v1 header) or every event before the last
//!   `session/end-seed` tagged `inherited: true` (v2+). That prefix still renders,
//!   but its usage is the parent's — already counted there — so it is not
//!   counted again.
//! - `compaction/prune` is a shadow-price row metering the node that the
//!   `surfaceOp: replace` event right after it rewrites; it is log-only.
//! - v4 lifts `tool/result` out of its `tool-result` wrapper block: the message
//!   has `role: "tool"`, the result blocks as its own `content`, and
//!   `toolCallId` / `isError` beside them.
//! - v2+ sessions never emit `assistant/chunk` rows: a model call that committed
//!   no surface message is recorded as `assistant/attempt`, whose embedded
//!   stream carries no surfaced text; its streamed usage chunk still counts
//!   (the call was billed).
//! - PTC (`run_code`) executes inner tool calls; each one is logged as a
//!   `tool/ptc-dispatch-start` + `tool/ptc-dispatch` pair (`tool/code-dispatch*`
//!   in v2) keyed by `subCallId`, and surfaces as its own Tool message after the
//!   `run_code` call that ran it.
//! - `system/message` carries the session's system prompt and v4 `developer/message`
//!   the tool-availability changes. Both are dropped like the other
//!   system-injected context dumps: harness context, not conversation.
//! - `deliverables/presented` lists files the harness presented to the user;
//!   it surfaces as a tagged System line following the subagent-report convention.
//! - A `turn/end` whose reason is not `completed` surfaces as a tagged status
//!   line (`[turn_failed]`, `[turn_cancelled]`, `[turn_interrupted]`,
//!   `[turn_blocked]`, `[turn_max_tokens]`, or `[turn_ended] <kind>` for a
//!   plugin-defined reason); a fork seed's `forked` closer is boundary
//!   bookkeeping and stays silent. Each `llm/retry` surfaces as a `[retry]`
//!   line (`llm/retry-started` only marks the wait ending).
//! - A user slash command is a `command/run` + `command/done` pair keyed by
//!   `commandId`: the run renders as command input (`/<name><args>`), the
//!   outcome text as command output.
//! - Delegation links: `subagent/catalog` (every delegated child, appended
//!   while the delegating tool runs) and `tool-workflow/agent-start` (a
//!   workflow agent's label) name the child session id. DSH runs one tool at a
//!   time, so the child belongs to the single open Agent-category tool call;
//!   its id and label land in that Tool message's
//!   `structured.childConversationIds` / `childPrompts`. Any other open-call
//!   count leaves the child unlinked (logged) — a format migration appends
//!   missing catalog facts after the final turn, where no call is open.
//! - Image blocks reference content-addressed attachments
//!   (`attachment.attachmentId: "sha256:<hex>"`) stored at
//!   `$DSH_HOME/attachments/v1/objects/<hex[..2]>/<hex>`; an existing object
//!   renders as `[Image: source: <path>]`, anything else as `[Image]`. A
//!   user-attached file block renders as `[File: <name>]`.
//! - Token usage also comes from `compaction/summary.data.usage` (the
//!   summarization call, attributed to its own `model`).
//! - Title precedence: a delegated session's descriptor label, then the latest
//!   non-fallback `session/title` (LLM or user-assigned), then the first user
//!   message, then DSH's `fallback` title (a few leading prompt words).
//! - Everything else in DSH's known event vocabulary (approval, feedback,
//!   hooks, teams, schedules, workflow lifecycle, image offload, delivery
//!   watermarks, …) is log-only.
//! - Every following record is a session event `{type, seq, time, data}` or a
//!   packed chunk row (`text-chunks` / `reasoning-chunks` / `tool-call-chunks`)
//!   that replays raw stream deltas in one storage line.
//! - The conversation surface comes from `user/message`, `assistant/message`,
//!   `tool/call` + `tool/result`, the PTC dispatch pairs, and the command
//!   pairs; the status lines, delegation links, titles and usage listed above
//!   are read from their own events. Every other event is log-only and never contributes to the
//!   transcript.
//!
//! Surface mapping:
//! - `user/message` whose `source.kind` is `"user"` becomes a User message.
//!   Workspace-instruction dumps (`source.kind == "agent-instructions"`) and
//!   any producer's context dump (`form` of `snapshot`, `instructions`,
//!   `catalog` or `recall`) are system-injected context, not conversation —
//!   the DSH GUI collapses them, so we drop them. A `session-reference`
//!   recall keeps only the referenced sessions' labels, as a
//!   `[session_reference]` line. Every other surfaced
//!   user-role message (approval-policy changes, goal rounds, relayed agent
//!   messages, cron notices, …) renders as a System line.
//! - `assistant/message` blocks: `text` → Assistant message (flushed around
//!   tool calls), `reasoning` → System `[thinking]` line (Claude convention),
//!   `tool-call` → Tool message with metadata, `image` → image marker.
//! - `tool/result` content is attached to the Tool message whose `callId` the
//!   `assistant/message` (or `tool/call`) surfaced; an orphan result creates a
//!   standalone Tool message.
//! - A compaction checkpoint (`user/message` from the `compact` producer)
//!   renders as a `[context_compacted]` System line carrying its summary.
//! - The creation-time `agentPreset` and later `agent-preset/selected` commits
//!   fold into `SessionMeta.variant_name`; the later committed selection wins.
//! - Usage (`assistant/message.usage`) is attached to the event's last
//!   non-System message, mirroring the Claude provider; a thinking-only step
//!   still gets a placeholder so accounting is never silently dropped.
//!
//! Robustness:
//! - A torn final record (no trailing newline) or a torn final zstd frame is a
//!   crash artifact; DSH's own scanner keeps the complete records before it, so
//!   do we (no parse warning). A read failure anywhere else keeps the records
//!   read so far and counts a parse warning.
//! - Malformed lines and unknown *required* event types (no `ignorable: true`)
//!   are logged and counted into `parse_warning_count` so the UI shows the ⚠
//!   badge instead of rendering a silently wrong transcript. Unknown ignorable
//!   rows are dropped silently, as the spec intends.
//! - A step whose stream never assembled an `assistant/message` (interrupted
//!   mid-stream) is reconstructed from its buffered chunks at `step/end`
//!   (or end of file), so crash-orphaned steps still show their partial text.
//! - The transcript keeps the complete history. A `surfaceOp: replace` event
//!   rewrites only the model's context: the nodes it shadows stay in place,
//!   its own messages appear where it was logged, and a replacement
//!   `tool/result` (an output trimmed for the model) never overwrites the
//!   result it shadows.
//!
//! Token usage rides `ParsedSession::usage_events` (one row per billed model
//! call: `assistant/message`, `assistant/attempt`, `compaction/summary`);
//! per-message `token_usage` is attached for display.
//! An event without its own model falls back to the latest routed request
//! model (`request/context`, `request/header`), then the session-level model;
//! a row that still lacks a model (or a timestamp) is skipped as a counted
//! parse warning, never silently.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::models::{Message, MessageRole, Provider, SessionMeta, TokenUsage};
use crate::provider::util::{UsageKeys, epoch_ms_to_rfc3339, session_title, token_usage_from};
use crate::provider::{ParsedSession, ProviderError, UsageEvent, system_time_to_epoch_seconds};
use crate::tool_metadata::{
    ToolCallFacts, ToolResultFacts, build_tool_metadata, enrich_tool_metadata,
};

/// DSH `assistant/message.usage` field paths (camelCase, disjoint counts).
const DSH_USAGE_KEYS: UsageKeys = UsageKeys {
    input: &["inputTokens"],
    output: &["outputTokens"],
    cache_read: &["cacheReadTokens"],
    cache_write: &["cacheWriteTokens"],
};

/// Highest session-header `version` whose format changes this parser handles
/// (see the module docs). Newer generations still parse best-effort but count
/// a warning so the UI badges them for review.
const MAX_SUPPORTED_HEADER_VERSION: i64 = 4;

const MISSING: Value = Value::Null;

#[derive(serde::Deserialize)]
struct DshHeader {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    version: Option<i64>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "createdAt")]
    created_at: Option<i64>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default, rename = "parentSession")]
    parent_session: Option<String>,
    #[serde(default)]
    origin: Option<String>,
    #[serde(default, rename = "agentPreset")]
    agent_preset: Option<String>,
    /// v0/v1 seeded sessions: how many leading events are inherited.
    #[serde(default, rename = "seedLength")]
    seed_length: Option<u64>,
}

/// Buffered stream deltas for one step whose `assistant/message` never
/// assembled (interrupted stream). Keyed by stream block index so blocks keep
/// their relative order when flushed.
#[derive(Default)]
struct StepChunkBuf {
    text: BTreeMap<usize, String>,
    reasoning: BTreeMap<usize, String>,
    /// Block index → (call id, tool name, joined raw arguments).
    tool_calls: BTreeMap<usize, (String, Option<String>, String)>,
    /// Last streamed usage chunk for the step. The assembled message's own
    /// usage supersedes it (the buffer is dropped on assembly), so this only
    /// feeds interrupted steps at flush — without it their tokens vanish.
    usage: Option<TokenUsage>,
    /// Timestamp of that usage chunk's row, for the usage event's bucket.
    usage_timestamp: Option<String>,
}

#[derive(Default)]
struct ParseState {
    messages: Vec<Message>,
    content_parts: Vec<String>,
    parse_warning_count: u32,
    /// `toolCallId` → index into `messages` for the surfaced Tool message.
    tool_by_call_id: HashMap<String, usize>,
    /// Call ids whose result is already shown; a replacement result for one
    /// of them (an output trimmed for the model) leaves it intact.
    tool_results: HashSet<String>,
    /// Latest LLM-generated or user-assigned `session/title`.
    latest_title: Option<String>,
    /// DSH's deterministic `fallback` title (a few leading prompt words).
    fallback_title: Option<String>,
    first_user_text: Option<String>,
    /// `subagent/descriptor.label`: the delegation name the parent chose.
    /// A delegated session is never LLM-titled, so this is its display name.
    descriptor_label: Option<String>,
    last_event_time_ms: Option<i64>,
    model: Option<String>,
    /// Effective agent preset. A committed `agent-preset/selected` event
    /// overrides the immutable creation-time header value.
    agent_preset: Option<String>,
    /// Chunk deltas per (turn, step), dropped once the step's
    /// `assistant/message` arrives or flushed at `step/end` when it never does.
    chunk_bufs: HashMap<(u32, u32), StepChunkBuf>,
    /// Steps whose `assistant/message` already assembled; late chunk rows for
    /// them are ignored so an out-of-order log cannot duplicate their text.
    assembled_steps: HashSet<(u32, u32)>,
    /// One row per `assistant/message` event that reports usage.
    usage_events: Vec<UsageEvent>,
    /// First seq after a v0/v1 header's inherited prefix, until reached.
    seed_cut: Option<u64>,
    /// `$DSH_HOME/attachments/v1/objects`, derived from the log's location.
    attachment_objects: Option<PathBuf>,
    /// Model of the latest routed request (`request/context`,
    /// `request/header`): what a usage row without its own model billed.
    request_model: Option<String>,
    /// Tool calls started (`tool/call`, PTC dispatch start), not yet settled.
    open_tool_calls: Vec<String>,
    /// Linked child session id → (Tool message index, position in that
    /// message's `childConversationIds`).
    child_links: HashMap<String, (usize, usize)>,
}

impl ParseState {
    fn push_user(&mut self, text: String, timestamp: Option<String>) {
        if self.first_user_text.is_none() {
            self.first_user_text = Some(text.clone());
        }
        self.content_parts.push(text.clone());
        self.messages.push(Message {
            timestamp,
            ..Message::user(text)
        });
    }

    fn push_assistant(&mut self, text: String, model: Option<String>, timestamp: Option<String>) {
        self.content_parts.push(text.clone());
        self.messages.push(Message {
            timestamp,
            model,
            ..Message::assistant(text)
        });
    }

    fn note_warning(&mut self) {
        self.parse_warning_count = self.parse_warning_count.saturating_add(1);
    }

    /// Record one billed model call. `model` falls back to the latest routed
    /// request model, then the session model; a row that still has no model
    /// (or no timestamp to bucket by) is skipped as a counted warning.
    fn push_usage_event(
        &mut self,
        usage: &TokenUsage,
        model: Option<String>,
        timestamp: Option<&str>,
    ) {
        let model = model
            .or_else(|| self.request_model.clone())
            .or_else(|| self.model.clone());
        match (timestamp, model) {
            (Some(timestamp), Some(model)) => self.usage_events.push(UsageEvent {
                timestamp: timestamp.to_string(),
                model,
                turn_count: 1,
                input_tokens: u64::from(usage.input_tokens),
                output_tokens: u64::from(usage.output_tokens),
                cache_read_input_tokens: u64::from(usage.cache_read_input_tokens),
                cache_creation_input_tokens: u64::from(usage.cache_creation_input_tokens),
                usage_hash: None,
                cost_is_estimate: false,
                cost_usd: None,
            }),
            _ => {
                log::warn!("skipping DSH usage without a timestamp or model");
                self.note_warning();
            }
        }
    }

    /// Everything parsed so far is the parent history a seeded session
    /// inherited. Its usage is counted on the parent, so drop it here —
    /// from the per-message display too, since a session without usage
    /// events falls back to message usage for its stats.
    fn drop_inherited_usage(&mut self) {
        self.usage_events.clear();
        for message in &mut self.messages {
            message.token_usage = None;
        }
        for buf in self.chunk_bufs.values_mut() {
            buf.usage = None;
        }
    }
}

/// Open a session artifact, transparently decompressing zstd frames when the
/// file name carries the `.zstd` suffix.
fn open_log_reader(path: &Path, file: File) -> Result<Box<dyn BufRead>, ProviderError> {
    let is_zstd = path.extension().and_then(|e| e.to_str()) == Some("zstd");
    if is_zstd {
        let decoder = zstd::stream::read::Decoder::new(BufReader::new(file))?;
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Consume every record after the header line, dispatching surface events
/// into `state`. Returns `None` when the file is unusable (unreadable stream,
/// missing/malformed header); a torn final record without a trailing newline
/// is ignored like DSH's own scanner does.
fn scan_records(
    mut reader: Box<dyn BufRead>,
    path: &Path,
    state: &mut ParseState,
) -> Option<DshHeader> {
    let mut buffer: Vec<u8> = Vec::new();
    let mut header: Option<DshHeader> = None;
    let mut line_no = 0usize;
    loop {
        buffer.clear();
        let n = match reader.read_until(b'\n', &mut buffer) {
            Ok(n) => n,
            // The final zstd frame was cut mid-write: a crash artifact. The
            // complete records before it stand, as in DSH's own reader.
            Err(error) if header.is_some() && error.kind() == ErrorKind::UnexpectedEof => {
                log::debug!("DSH session '{}' ends in a torn frame", path.display());
                break;
            }
            Err(error) => {
                log::warn!("failed to read DSH session '{}': {error}", path.display());
                header.as_ref()?;
                // Keep what was read, flagged: the rest is unreadable.
                state.note_warning();
                break;
            }
        };
        if n == 0 {
            break;
        }
        if buffer.last() != Some(&b'\n') {
            // Torn tail: the log was cut mid-write. Not a parse warning.
            break;
        }
        line_no += 1;
        let line = match std::str::from_utf8(&buffer[..buffer.len() - 1]) {
            Ok(line) => line,
            Err(error) => {
                log::warn!(
                    "skipping non-UTF-8 DSH record at line {line_no} in '{}': {error}",
                    path.display()
                );
                state.note_warning();
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        if header.is_none() {
            match serde_json::from_str::<DshHeader>(line) {
                Ok(parsed) if parsed.kind == "session" => {
                    if parsed
                        .version
                        .is_some_and(|v| v > MAX_SUPPORTED_HEADER_VERSION)
                    {
                        log::warn!(
                            "DSH session '{}' has format version {:?}; tested up to {MAX_SUPPORTED_HEADER_VERSION} — parsing best-effort",
                            path.display(),
                            parsed.version
                        );
                        state.note_warning();
                    }
                    state.seed_cut = parsed.seed_length.filter(|length| *length > 0);
                    header = Some(parsed);
                }
                Ok(parsed) => {
                    log::warn!(
                        "first DSH record of '{}' is not a session header (type '{}')",
                        path.display(),
                        parsed.kind
                    );
                    return None;
                }
                Err(error) => {
                    log::warn!(
                        "malformed DSH header at line {line_no} in '{}': {error}",
                        path.display()
                    );
                    return None;
                }
            }
            continue;
        }
        let record: Value = match serde_json::from_str(line) {
            Ok(record) => record,
            Err(error) => {
                log::warn!(
                    "skipping malformed DSH record at line {line_no} in '{}': {error}",
                    path.display()
                );
                state.note_warning();
                continue;
            }
        };
        handle_record(&record, state);
    }
    header
}

fn handle_record(record: &Value, state: &mut ParseState) {
    let Some(event_type) = record.get("type").and_then(Value::as_str) else {
        log::warn!("skipping DSH record without a type tag");
        state.note_warning();
        return;
    };
    if let Some(time) = record.get("time").and_then(Value::as_i64) {
        state.last_event_time_ms = Some(time);
    }
    let timestamp = record
        .get("time")
        .and_then(Value::as_i64)
        .and_then(epoch_ms_to_rfc3339);
    let data = record.get("data").unwrap_or(&MISSING);
    let seq = record.get("seq").and_then(Value::as_u64);
    if let (Some(cut), Some(seq)) = (state.seed_cut, seq)
        && seq >= cut
    {
        state.seed_cut = None;
        state.drop_inherited_usage();
    }
    match event_type {
        "session/title" => {
            if let Some(title) = data.get("title").and_then(Value::as_str)
                && !title.trim().is_empty()
            {
                if data.pointer("/source/kind").and_then(Value::as_str) == Some("fallback") {
                    state.fallback_title = Some(title.to_string());
                } else {
                    state.latest_title = Some(title.to_string());
                }
            }
        }
        "turn/end" => handle_turn_end(data, state, timestamp),
        "command/run" => handle_command_run(data, state, timestamp),
        "command/done" => handle_command_done(data, state, timestamp),
        "llm/retry" => handle_llm_retry(data, state, timestamp),
        // The routed model of the request that follows; a usage row without
        // its own model was billed against it.
        "request/context" => {
            if let Some(model) = data.get("model").and_then(Value::as_str) {
                state.request_model = Some(model.to_string());
            }
        }
        "request/header" => {
            if let Some(model) = data.pointer("/header/config/model").and_then(Value::as_str) {
                state.request_model = Some(model.to_string());
            }
        }
        // The summarization call that produced a compaction checkpoint.
        "compaction/summary" => {
            if let Some(usage) = data
                .get("usage")
                .and_then(|usage| token_usage_from(usage, &DSH_USAGE_KEYS))
            {
                let model = data
                    .get("model")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                state.push_usage_event(&usage, model, timestamp.as_deref());
            }
        }
        // A model call that committed no surface message; its streamed usage
        // (when the provider reported any) was still billed.
        "assistant/attempt" => {
            if let Some(usage) = last_stream_usage(data).filter(|usage| usage.total_tokens() > 0) {
                state.push_usage_event(&usage, None, timestamp.as_deref());
            }
        }
        "subagent/catalog" => {
            if let Some(child_id) = data.get("childId").and_then(Value::as_str) {
                let label = data.get("label").and_then(Value::as_str);
                link_child_session(state, child_id, label);
            }
        }
        "tool-workflow/agent-start" => {
            if let Some(child_id) = data.get("childId").and_then(Value::as_str) {
                let label = data.get("label").and_then(Value::as_str);
                link_child_session(state, child_id, label);
            }
        }
        "tool/ptc-dispatch-start" | "tool/code-dispatch-start" => {
            handle_ptc_dispatch_start(data, state, timestamp);
        }
        "tool/ptc-dispatch" | "tool/code-dispatch" => {
            handle_ptc_dispatch(data, state, timestamp);
        }
        "user/message" => handle_user_message(data, state, timestamp),
        "assistant/message" => handle_assistant_message(data, state, timestamp),
        "tool/call" => handle_tool_call(data, state, timestamp),
        "tool/result" => {
            let replaces =
                record.pointer("/surfaceOp/op").and_then(Value::as_str) == Some("replace");
            handle_tool_result(data, state, timestamp, replaces);
        }
        "assistant/chunk" | "text-chunks" | "reasoning-chunks" | "tool-call-chunks" => {
            handle_chunk_event(event_type, data, state, timestamp);
        }
        // A delegated (subagent) session opens with its descriptor; the
        // `label` is the delegation name the parent chose — the only
        // human-meaningful title such a session ever gets.
        "subagent/descriptor" => {
            if let Some(label) = data
                .get("label")
                .and_then(Value::as_str)
                .filter(|label| !label.trim().is_empty())
            {
                state.descriptor_label = Some(label.to_string());
            }
        }
        // The system prompt and v4 tool-availability changes arrive as
        // surface events, but they are harness context rather than
        // conversation — dropped like the context dumps in user/message.
        "system/message" | "developer/message" => {
            log::debug!("skipping DSH {event_type} context");
        }
        // The last marker tagged `inherited` ends a v2+ seeded session's
        // copy of its parent's history; untagged markers carry no cut.
        "session/end-seed" => {
            if data.get("inherited").and_then(Value::as_bool) == Some(true) {
                state.drop_inherited_usage();
            }
        }
        // Files the harness presented to the user: surface compactly as a
        // tagged System line (subagent-report convention), not conversation.
        "deliverables/presented" => {
            let files: Vec<String> = data
                .get("files")
                .and_then(Value::as_array)
                .map(|files| {
                    files
                        .iter()
                        .filter_map(|file| {
                            let path = file.get("path").and_then(Value::as_str)?;
                            let description = file
                                .get("description")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .trim();
                            if description.is_empty() {
                                Some(path.to_string())
                            } else {
                                Some(format!("{path} — {description}"))
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            if !files.is_empty() {
                state.messages.push(Message {
                    timestamp,
                    ..Message::system(format!("[deliverables]\n{}", files.join("\n")))
                });
            }
        }
        "agent-preset/selected" => {
            if let Some(agent_preset) = data
                .get("agentPreset")
                .and_then(Value::as_str)
                .filter(|name| !name.trim().is_empty())
            {
                state.agent_preset = Some(agent_preset.to_string());
            } else {
                log::warn!("skipping malformed DSH agent-preset/selected event");
                state.note_warning();
            }
        }
        "step/end" => {
            if let (Some(turn), Some(step)) = (
                data.get("turn").and_then(Value::as_u64),
                data.get("step").and_then(Value::as_u64),
            ) {
                flush_step_chunks(state, turn as u32, step as u32);
            }
        }
        // Known log-only event families: boundaries, chunk rows, compaction
        // brackets and shadow prices, approvals, feedback, hooks, teams,
        // schedules, workflow lifecycle, model-facing image offload, and
        // transport/model bookkeeping. None carries surface semantics.
        "turn/start"
        | "step/start"
        | "todo/write"
        | "permission/preset"
        | "sandbox/mode"
        | "approval/policy"
        | "agent/inbox/spliced"
        | "session/title-llm-request"
        | "llm/retry-started"
        | "compaction/start"
        | "compaction/end"
        | "compaction/prune"
        | "image/offload"
        | "approval/requested"
        | "approval/resolved"
        | "approval/asked"
        | "approval/decided"
        | "feedback/record"
        | "feedback/message-put"
        | "feedback/message-delete"
        | "hook/invoked"
        | "hook/result"
        | "team/member"
        | "team/task"
        | "team/message/queued"
        | "team/message/delivered"
        | "schedule/change"
        | "subagent/model-selection-policy"
        | "tool-workflow/run-start"
        | "tool-workflow/run-end"
        | "tool-workflow/agent-end"
        | "workspace/changes"
        | "model/selection"
        | "session-log-deepseek/delivery-accepted"
        | "activity/status"
        | "goal/change"
        | "plan/mode"
        | "question/requested"
        | "question/resolved"
        | "stream/error"
        | "session/created"
        | "session/event"
        | "web/deepseek-search-llm-request" => {}
        unknown => {
            let ignorable = record
                .get("ignorable")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if ignorable {
                log::debug!("skipping ignorable DSH event '{unknown}'");
            } else {
                // A required event we cannot interpret may change the meaning
                // of the surrounding log; surface it as a parse warning rather
                // than rendering a silently wrong transcript.
                log::warn!("skipping unknown required DSH event '{unknown}'");
                state.note_warning();
            }
        }
    }
}

/// The transcript marker for an image block: `[Image: source: <path>]` when
/// its `sha256:` attachment object exists under `objects`, else `[Image]`.
/// Only a 64-digit hex id becomes a path, so a log cannot point elsewhere.
fn image_marker(block: &Value, objects: Option<&Path>) -> String {
    let path = block
        .pointer("/attachment/attachmentId")
        .and_then(Value::as_str)
        .and_then(|id| id.strip_prefix("sha256:"))
        .filter(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .zip(objects)
        .map(|(hex, objects)| objects.join(&hex[..2]).join(hex));
    match path {
        Some(path) if path.is_file() => format!("[Image: source: {}]", path.display()),
        Some(path) => {
            log::debug!("DSH image attachment missing: {}", path.display());
            "[Image]".to_string()
        }
        None => "[Image]".to_string(),
    }
}

/// A user-attached file: `[File: <name>]`, or `[File]` when unnamed.
fn file_marker(block: &Value) -> String {
    match block
        .pointer("/attachment/name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
    {
        Some(name) => format!("[File: {name}]"),
        None => "[File]".to_string(),
    }
}

/// Extract plain text from a message `content` payload, following the Claude
/// provider's convention of trailing attachment markers (images, files). The
/// payload is normally a `ContentBlock[]` array; a bare string (older or
/// foreign producers) is used verbatim.
fn extract_block_text(content: &Value, objects: Option<&Path>) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    let mut parts = Vec::new();
    let mut attachments = Vec::new();
    if let Some(blocks) = content.as_array() {
        for block in blocks {
            match block.get("type").and_then(Value::as_str).unwrap_or("") {
                "text" => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        parts.push(text.to_string());
                    }
                }
                "image" => attachments.push(image_marker(block, objects)),
                "file" => attachments.push(file_marker(block)),
                other => {
                    log::debug!("skipping DSH content block '{other}'");
                }
            }
        }
    }
    parts.extend(attachments);
    parts.join("\n")
}

fn handle_user_message(data: &Value, state: &mut ParseState, timestamp: Option<String>) {
    let source_kind = data
        .pointer("/source/kind")
        .and_then(Value::as_str)
        .unwrap_or("");
    let form = data.pointer("/source/form").and_then(Value::as_str);
    // v4 names the compaction producer `compact-checkpoint`; earlier versions
    // attribute it to the `compact` plugin.
    let is_checkpoint = source_kind == "compact-checkpoint"
        || (source_kind == "plugin"
            && data.pointer("/source/plugin").and_then(Value::as_str) == Some("compact"));
    let text = data
        .get("content")
        .map(|content| extract_block_text(content, state.attachment_objects.as_deref()))
        .unwrap_or_default();
    if text.trim().is_empty() {
        return;
    }
    match source_kind {
        // Direct human prompts are the user's own words.
        "user" => state.push_user(text, timestamp),
        // System-injected context dumps — workspace instructions, runtime
        // snapshots, skill/tool catalogs, relayed or recalled context — are
        // not conversation; the DSH UI collapses them, so do we. The dump
        // shape is identified by its `form`, not the producer `kind`
        // (`plugin`, `runtime-context`, `skill-catalog`, … all qualify).
        // A cross-session recall: the referenced sessions' captured history
        // is context for the model, but which sessions the user referenced is
        // part of the conversation — name them, drop the snapshot.
        "session-reference" => {
            let labels: Vec<&str> = data
                .pointer("/source/references")
                .and_then(Value::as_array)
                .map(|references| {
                    references
                        .iter()
                        .filter_map(|reference| {
                            reference
                                .get("label")
                                .and_then(Value::as_str)
                                .filter(|label| !label.trim().is_empty())
                                .or_else(|| reference.get("sessionId").and_then(Value::as_str))
                        })
                        .collect()
                })
                .unwrap_or_default();
            if labels.is_empty() {
                log::warn!("skipping DSH session reference without referenced sessions");
                state.note_warning();
            } else {
                state.messages.push(Message {
                    timestamp,
                    ..Message::system(format!("[session_reference]\n{}", labels.join("\n")))
                });
            }
        }
        "agent-instructions" => {
            log::debug!("skipping DSH agent-instructions user message");
        }
        // A background subagent's relayed report. The first block is DSH's
        // boilerplate ("Background subagent <id> reported:") — drop it and
        // render the report body as a tagged, collapsible system row.
        "subagent-report" => {
            let body = match text.split_once('\n') {
                Some((first, rest)) if first.starts_with("Background subagent") => rest.trim(),
                _ => text.trim(),
            };
            if !body.is_empty() {
                state.messages.push(Message {
                    timestamp,
                    ..Message::system(format!("[subagent_report] {body}"))
                });
            }
        }
        // The settle notice: boilerplate ("Background subagent <id>
        // finished…" + "Its closing message:") wraps the child's closing
        // report — strip the wrapper, tag the report.
        "subagent-settled" => {
            let mut body = text.as_str();
            if let Some((first, rest)) = body.split_once('\n')
                && first.starts_with("Background subagent")
            {
                body = rest;
            }
            let body = body.trim_start();
            let body = body
                .strip_prefix("Its closing message:")
                .unwrap_or(body)
                .trim();
            if !body.is_empty() {
                state.messages.push(Message {
                    timestamp,
                    ..Message::system(format!("[subagent_settled] {body}"))
                });
            }
        }
        // A compaction checkpoint: DSH frames the summary between
        // `<compacted-summary>` tags behind a model-facing preamble; show
        // just the summary under the shared compaction marker.
        _ if is_checkpoint => {
            let summary = text
                .split_once("<compacted-summary>")
                .and_then(|(_, rest)| rest.split_once("</compacted-summary>"))
                .map_or(text.as_str(), |(summary, _)| summary)
                .trim();
            state.messages.push(Message {
                timestamp,
                ..Message::system(format!("[context_compacted]\n{summary}"))
            });
        }
        // System-injected context dumps — workspace instructions, runtime
        // snapshots, skill/tool catalogs, recalled context — are not
        // conversation; the DSH UI collapses them, so do we. The dump shape is
        // identified by its `form`, not the producer `kind` (`plugin`,
        // `runtime-context`, `skill-catalog`, … all qualify).
        // `relay` is deliberately absent: `subagent-report` (handled above) and
        // `coordinator` both relay real instructions, which are conversation.
        _ if matches!(
            form,
            Some("snapshot") | Some("instructions") | Some("catalog") | Some("recall")
        ) =>
        {
            log::debug!("skipping DSH {source_kind} {form:?} context dump");
        }
        // Everything else that reaches the surface (policy changes, goal
        // rounds, notices, compaction checkpoints, …) renders as a system
        // line.
        _ => {
            state.messages.push(Message {
                timestamp,
                ..Message::system(text)
            });
        }
    }
}

/// Parse a raw arguments JSON string into a `Value` for tool metadata, or
/// `None` when the model produced malformed JSON.
fn parse_tool_arguments(raw: &str) -> Option<Value> {
    serde_json::from_str::<Value>(raw).ok()
}

fn push_tool_message(
    state: &mut ParseState,
    raw_name: &str,
    arguments_raw: &str,
    call_id: Option<&str>,
    timestamp: Option<String>,
) {
    let metadata = build_tool_metadata(ToolCallFacts {
        provider: Provider::Dsh,
        raw_name,
        input: parse_tool_arguments(arguments_raw).as_ref(),
        call_id,
        assistant_id: None,
    });
    let canonical_name = metadata.canonical_name.clone();
    let idx = state.messages.len();
    state.messages.push(Message {
        timestamp,
        tool_name: Some(canonical_name),
        tool_input: Some(arguments_raw.to_string()),
        tool_metadata: Some(metadata),
        ..Message::new(MessageRole::Tool, String::new())
    });
    if let Some(call_id) = call_id {
        state.tool_by_call_id.insert(call_id.to_string(), idx);
    }
}

fn handle_assistant_message(data: &Value, state: &mut ParseState, timestamp: Option<String>) {
    let Some(message) = data.get("message") else {
        return;
    };
    let model = message
        .pointer("/source/model")
        .and_then(Value::as_str)
        .map(str::to_string);
    if state.model.is_none() {
        state.model = model.clone();
    }
    let usage = data
        .get("usage")
        .and_then(|usage| token_usage_from(usage, &DSH_USAGE_KEYS));
    let turn_start = state.messages.len();
    let mut text_parts: Vec<String> = Vec::new();
    let mut images: Vec<String> = Vec::new();
    if let Some(blocks) = message.get("content").and_then(Value::as_array) {
        for block in blocks {
            match block.get("type").and_then(Value::as_str).unwrap_or("") {
                "reasoning" => {
                    if let Some(text) = block.get("text").and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        state.messages.push(Message {
                            timestamp: timestamp.clone(),
                            ..Message::system(format!("[thinking]\n{text}"))
                        });
                    }
                }
                "text" => {
                    if let Some(text) = block.get("text").and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        text_parts.push(text.to_string());
                    }
                }
                "tool-call" => {
                    if !text_parts.is_empty() {
                        let text = text_parts.join("\n");
                        state.push_assistant(text, model.clone(), timestamp.clone());
                        text_parts.clear();
                    }
                    let name = block.get("name").and_then(Value::as_str).unwrap_or("tool");
                    let call_id = block.get("id").and_then(Value::as_str);
                    // A tool/call event may already have surfaced this call
                    // (out-of-order log); don't duplicate the Tool message.
                    if call_id.is_some_and(|id| state.tool_by_call_id.contains_key(id)) {
                        continue;
                    }
                    let arguments_raw =
                        block.get("arguments").and_then(Value::as_str).unwrap_or("");
                    push_tool_message(state, name, arguments_raw, call_id, timestamp.clone());
                }
                "image" => images.push(image_marker(block, state.attachment_objects.as_deref())),
                other => {
                    log::warn!("skipping unknown DSH assistant content block '{other}'");
                    state.note_warning();
                }
            }
        }
    }
    text_parts.extend(images);
    if !text_parts.is_empty() {
        state.push_assistant(text_parts.join("\n"), model.clone(), timestamp.clone());
    }
    if let Some(usage) = usage {
        state.push_usage_event(&usage, model.clone(), timestamp.as_deref());
        // Attach usage to the event's last non-System message so accounting
        // is never silently dropped; a thinking-only step gets a placeholder.
        if let Some(last) = state.messages[turn_start..]
            .iter_mut()
            .filter(|message| message.role != MessageRole::System)
            .last()
        {
            last.token_usage = Some(usage);
            if last.model.as_deref().map(str::is_empty).unwrap_or(true) {
                last.model = model.clone();
            }
            if last.timestamp.is_none() {
                last.timestamp = timestamp.clone();
            }
        } else {
            state.messages.push(Message {
                timestamp: timestamp.clone(),
                token_usage: Some(usage),
                model: model.clone(),
                ..Message::assistant(String::new())
            });
        }
    }
    // The assembled message supersedes the step's raw chunks.
    if let (Some(turn), Some(step)) = (
        data.get("turn").and_then(Value::as_u64),
        data.get("step").and_then(Value::as_u64),
    ) {
        let key = (turn as u32, step as u32);
        state.chunk_bufs.remove(&key);
        state.assembled_steps.insert(key);
    }
}

fn handle_tool_call(data: &Value, state: &mut ParseState, timestamp: Option<String>) {
    let Some(call_id) = data.get("callId").and_then(Value::as_str) else {
        return;
    };
    state.open_tool_calls.push(call_id.to_string());
    if state.tool_by_call_id.contains_key(call_id) {
        // Already surfaced through the step's assistant/message blocks.
        return;
    }
    let name = data.get("name").and_then(Value::as_str).unwrap_or("tool");
    let arguments_raw = data.get("arguments").and_then(Value::as_str).unwrap_or("");
    push_tool_message(state, name, arguments_raw, Some(call_id), timestamp);
}

/// `replaces`: the record is a surface replacement of an earlier result.
fn handle_tool_result(
    data: &Value,
    state: &mut ParseState,
    timestamp: Option<String>,
    replaces: bool,
) {
    let Some(message) = data.get("message") else {
        return;
    };
    // v4 results are `role: "tool"` messages carrying the result directly;
    // earlier versions wrap it in the message's single `tool-result` block.
    let result = if message.get("role").and_then(Value::as_str) == Some("tool") {
        message
    } else {
        message.pointer("/content/0").unwrap_or(&MISSING)
    };
    let call_id = message
        .pointer("/source/callId")
        .and_then(Value::as_str)
        .or_else(|| result.get("toolCallId").and_then(Value::as_str));
    if let Some(call_id) = call_id
        && !state.tool_results.insert(call_id.to_string())
        && replaces
    {
        return;
    }
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let result_text = result
        .get("content")
        .map(|content| extract_block_text(content, state.attachment_objects.as_deref()))
        .unwrap_or_default();
    settle_tool_call(
        state,
        call_id,
        result_text,
        is_error,
        timestamp,
        OrphanCall::default(),
    );
}

/// What an orphan result knows about the call it settles.
#[derive(Default)]
struct OrphanCall<'a> {
    name: Option<&'a str>,
    arguments: Option<String>,
}

/// Attach a result to the Tool message its `call_id` surfaced and close the
/// call. A result whose call never surfaced (e.g. the model call was
/// interrupted before its message assembled) becomes a standalone Tool
/// message built from `orphan`.
fn settle_tool_call(
    state: &mut ParseState,
    call_id: Option<&str>,
    result_text: String,
    is_error: bool,
    timestamp: Option<String>,
    orphan: OrphanCall<'_>,
) {
    if let Some(call_id) = call_id {
        state.open_tool_calls.retain(|open| open != call_id);
    }
    let result_facts = ToolResultFacts {
        is_error: Some(is_error),
        ..ToolResultFacts::default()
    };
    if let Some(call_id) = call_id
        && let Some(&idx) = state.tool_by_call_id.get(call_id)
    {
        if !result_text.is_empty() {
            state.messages[idx].content = result_text;
        }
        if let Some(metadata) = state.messages[idx].tool_metadata.as_mut() {
            enrich_tool_metadata(metadata, result_facts);
        }
        return;
    }
    let input = orphan.arguments.as_deref().and_then(parse_tool_arguments);
    let mut metadata = build_tool_metadata(ToolCallFacts {
        provider: Provider::Dsh,
        raw_name: orphan.name.unwrap_or("tool"),
        input: input.as_ref(),
        call_id,
        assistant_id: None,
    });
    enrich_tool_metadata(&mut metadata, result_facts);
    let canonical_name = metadata.canonical_name.clone();
    state.messages.push(Message {
        timestamp,
        tool_name: Some(canonical_name),
        tool_input: orphan.arguments,
        tool_metadata: Some(metadata),
        ..Message::new(MessageRole::Tool, result_text)
    });
}

/// A PTC sub-call started inside `run_code`: its own Tool message, keyed by
/// `subCallId`. Arguments are logged as a JSON object, not a raw string.
fn handle_ptc_dispatch_start(data: &Value, state: &mut ParseState, timestamp: Option<String>) {
    let Some(sub_call_id) = data.get("subCallId").and_then(Value::as_str) else {
        log::warn!("skipping DSH PTC dispatch start without a subCallId");
        state.note_warning();
        return;
    };
    state.open_tool_calls.push(sub_call_id.to_string());
    if state.tool_by_call_id.contains_key(sub_call_id) {
        return;
    }
    let name = data.get("name").and_then(Value::as_str).unwrap_or("tool");
    let arguments = data
        .get("arguments")
        .map(Value::to_string)
        .unwrap_or_default();
    push_tool_message(state, name, &arguments, Some(sub_call_id), timestamp);
}

/// A settled PTC sub-call: attach its result to the sub-call's Tool message.
fn handle_ptc_dispatch(data: &Value, state: &mut ParseState, timestamp: Option<String>) {
    let Some(sub_call_id) = data.get("subCallId").and_then(Value::as_str) else {
        log::warn!("skipping DSH PTC dispatch without a subCallId");
        state.note_warning();
        return;
    };
    let is_error = data
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let result_text = data
        .get("content")
        .map(|content| extract_block_text(content, state.attachment_objects.as_deref()))
        .unwrap_or_default();
    let orphan = OrphanCall {
        name: data.get("name").and_then(Value::as_str),
        arguments: data.get("arguments").map(Value::to_string),
    };
    settle_tool_call(
        state,
        Some(sub_call_id),
        result_text,
        is_error,
        timestamp,
        orphan,
    );
}

/// Link a delegated child session to the Tool message that spawned it: the
/// single open Agent-category call (DSH runs one tool at a time). A known
/// child only refreshes its label — `tool-workflow/agent-start` names a
/// workflow agent its catalog entry left unlabeled.
fn link_child_session(state: &mut ParseState, child_id: &str, label: Option<&str>) {
    let label = label.map(str::trim).filter(|label| !label.is_empty());
    if let Some(&(idx, position)) = state.child_links.get(child_id) {
        if let Some(label) = label
            && let Some(prompts) = state.messages[idx]
                .tool_metadata
                .as_mut()
                .and_then(|metadata| metadata.structured.as_mut())
                .and_then(|structured| structured.get_mut("childPrompts"))
                .and_then(Value::as_array_mut)
            && let Some(slot) = prompts.get_mut(position)
        {
            *slot = Value::String(label.to_string());
        }
        return;
    }
    let candidates: Vec<usize> = state
        .open_tool_calls
        .iter()
        .filter_map(|call_id| state.tool_by_call_id.get(call_id).copied())
        .filter(|&idx| state.messages[idx].tool_name.as_deref() == Some("Agent"))
        .collect();
    let idx = match candidates[..] {
        [idx] => idx,
        // A format migration appends missing catalog facts after the final
        // turn, where no call is open; those children stay unlinked.
        [] => {
            log::debug!("DSH child session {child_id} has no open delegating tool call");
            return;
        }
        _ => {
            log::warn!(
                "DSH child session {child_id} has {} open delegating tool calls; leaving it unlinked",
                candidates.len()
            );
            return;
        }
    };
    let Some(metadata) = state.messages[idx].tool_metadata.as_mut() else {
        return;
    };
    let structured = metadata
        .structured
        .get_or_insert_with(|| Value::Object(serde_json::Map::new()));
    let Some(object) = structured.as_object_mut() else {
        log::warn!("DSH delegating tool metadata is not an object; leaving {child_id} unlinked");
        return;
    };
    let Some(ids) = object
        .entry("childConversationIds")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
    else {
        return;
    };
    ids.push(Value::String(child_id.to_string()));
    let position = ids.len() - 1;
    let Some(prompts) = object
        .entry("childPrompts")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
    else {
        return;
    };
    // Prompts stay positionally aligned with the ids.
    prompts.resize(position, Value::String(String::new()));
    prompts.push(Value::String(label.unwrap_or("").to_string()));
    state
        .child_links
        .insert(child_id.to_string(), (idx, position));
}

/// The last usage chunk a model attempt streamed.
fn last_stream_usage(data: &Value) -> Option<TokenUsage> {
    data.get("stream")?
        .as_array()?
        .iter()
        .rev()
        .filter_map(|record| record.get("chunk"))
        .filter(|chunk| chunk.get("type").and_then(Value::as_str) == Some("usage"))
        .find_map(|chunk| {
            chunk
                .get("usage")
                .and_then(|usage| token_usage_from(usage, &DSH_USAGE_KEYS))
        })
}

/// A user slash command (`/compact`, `/plan …`, a permission switch, …) as
/// the command line the user issued. `args` is the verbatim raw input,
/// separator included; it is absent when a domain event owns the input.
fn handle_command_run(data: &Value, state: &mut ParseState, timestamp: Option<String>) {
    let Some(name) = data
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
    else {
        log::warn!("skipping DSH command/run without a command name");
        state.note_warning();
        return;
    };
    let args = data.get("args").and_then(Value::as_str).unwrap_or("");
    let line = format!("/{name}{args}").trim_end().to_string();
    state.content_parts.push(line.clone());
    state.messages.push(Message {
        timestamp,
        ..Message::command_input(line)
    });
}

/// The settled command's verbatim outcome text, as command output.
fn handle_command_done(data: &Value, state: &mut ParseState, timestamp: Option<String>) {
    if data.get("kind").and_then(Value::as_str).is_none() {
        log::warn!("skipping DSH command/done without an outcome kind");
        state.note_warning();
        return;
    }
    let Some(text) = data
        .get("text")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
    else {
        return;
    };
    state.content_parts.push(text.to_string());
    state.messages.push(Message {
        timestamp,
        ..Message::command_output(text)
    });
}

/// A model request retry: `[retry] retry <n>/<max> after <ms> ms (<code>)`
/// plus the failure message (`always` mode has no retry ceiling, so no
/// `/<max>`).
fn handle_llm_retry(data: &Value, state: &mut ParseState, timestamp: Option<String>) {
    let (Some(retry), Some(delay_ms)) = (
        data.get("retry").and_then(Value::as_u64),
        data.get("delayMs").and_then(Value::as_f64),
    ) else {
        log::warn!("skipping DSH llm/retry without retry details");
        state.note_warning();
        return;
    };
    let attempt = match data.get("maxRetries").and_then(Value::as_u64) {
        Some(max) => format!("{retry}/{max}"),
        None => retry.to_string(),
    };
    let mut summary = format!("retry {attempt} after {} ms", delay_ms.round());
    if let Some(code) = data.pointer("/failure/code").and_then(Value::as_str) {
        summary.push_str(&format!(" ({code})"));
    }
    let content = match data
        .pointer("/failure/message")
        .and_then(Value::as_str)
        .filter(|message| !message.trim().is_empty())
    {
        Some(message) => format!("[retry] {summary}\n{message}"),
        None => format!("[retry] {summary}"),
    };
    state.messages.push(Message {
        timestamp,
        ..Message::system(content)
    });
}

/// A turn that ended other than `completed` surfaces as a tagged status line.
fn handle_turn_end(data: &Value, state: &mut ParseState, timestamp: Option<String>) {
    let reason = data.get("reason").unwrap_or(&MISSING);
    let Some(kind) = reason.get("kind").and_then(Value::as_str) else {
        log::warn!("skipping DSH turn/end without a reason kind");
        state.note_warning();
        return;
    };
    let content = match kind {
        // A fork seed's closer marks the copy boundary, not an outcome.
        "completed" | "forked" => return,
        "error" => {
            let message = reason
                .pointer("/error/message")
                .and_then(Value::as_str)
                .filter(|message| !message.trim().is_empty());
            message.map_or_else(
                || "[turn_failed]".to_string(),
                |message| format!("[turn_failed]\n{message}"),
            )
        }
        "aborted" => match reason.pointer("/reason/kind").and_then(Value::as_str) {
            Some("hook") => {
                let detail = reason
                    .pointer("/reason/reason")
                    .and_then(Value::as_str)
                    .unwrap_or("hook");
                format!("[turn_cancelled] hook: {detail}")
            }
            Some(cause @ ("parent" | "disposed")) => format!("[turn_cancelled] {cause}"),
            _ => "[turn_cancelled]".to_string(),
        },
        "interrupted" => "[turn_interrupted]".to_string(),
        "blocked" => "[turn_blocked]".to_string(),
        "max-tokens" => "[turn_max_tokens]".to_string(),
        other => format!("[turn_ended] {other}"),
    };
    state.messages.push(Message {
        timestamp,
        ..Message::system(content)
    });
}

fn handle_chunk_event(
    event_type: &str,
    data: &Value,
    state: &mut ParseState,
    timestamp: Option<String>,
) {
    let (Some(turn), Some(step)) = (
        data.get("turn").and_then(Value::as_u64),
        data.get("step").and_then(Value::as_u64),
    ) else {
        return;
    };
    let key = (turn as u32, step as u32);
    if state.assembled_steps.contains(&key) {
        // The step's assistant/message already assembled; its chunks were
        // superseded, so an out-of-order chunk row must not re-buffer text.
        return;
    }
    let buf = state.chunk_bufs.entry(key).or_default();
    match event_type {
        "assistant/chunk" => {
            let Some(chunk) = data.get("chunk") else {
                return;
            };
            // Usage chunks carry no block index — capture them before the
            // index gate, so an interrupted step keeps its token accounting.
            if chunk.get("type").and_then(Value::as_str) == Some("usage") {
                if let Some(usage) = chunk
                    .get("usage")
                    .and_then(|usage| token_usage_from(usage, &DSH_USAGE_KEYS))
                {
                    buf.usage = Some(usage);
                    if timestamp.is_some() {
                        buf.usage_timestamp = timestamp;
                    }
                }
                return;
            }
            let Some(index) = chunk.get("index").and_then(Value::as_u64) else {
                return;
            };
            let index = index as usize;
            match chunk.get("type").and_then(Value::as_str).unwrap_or("") {
                "text-delta" => {
                    if let Some(text) = chunk.get("text").and_then(Value::as_str) {
                        buf.text.entry(index).or_default().push_str(text);
                    }
                }
                "reasoning-delta" => {
                    if let Some(text) = chunk.get("text").and_then(Value::as_str) {
                        buf.reasoning.entry(index).or_default().push_str(text);
                    }
                }
                "tool-call-delta" => {
                    let id = chunk
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let name = chunk
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    let arguments_delta = chunk
                        .get("argumentsDelta")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let entry = buf
                        .tool_calls
                        .entry(index)
                        .or_insert_with(|| (id.clone(), name.clone(), String::new()));
                    entry.0 = id;
                    if name.is_some() {
                        entry.1 = name;
                    }
                    entry.2.push_str(arguments_delta);
                }
                _ => {}
            }
        }
        "text-chunks" | "reasoning-chunks" => {
            let Some(index) = data.get("index").and_then(Value::as_u64) else {
                return;
            };
            let joined: String = data
                .get("texts")
                .and_then(Value::as_array)
                .map(|texts| texts.iter().filter_map(Value::as_str).collect::<String>())
                .unwrap_or_default();
            if joined.is_empty() {
                return;
            }
            let index = index as usize;
            let target = if event_type == "text-chunks" {
                &mut buf.text
            } else {
                &mut buf.reasoning
            };
            target.entry(index).or_default().push_str(&joined);
        }
        "tool-call-chunks" => {
            let Some(index) = data.get("index").and_then(Value::as_u64) else {
                return;
            };
            let id = data
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let name = data.get("name").and_then(Value::as_str).map(str::to_string);
            let args: String = data
                .get("args")
                .and_then(Value::as_array)
                .map(|args| args.iter().filter_map(Value::as_str).collect::<String>())
                .unwrap_or_default();
            let index = index as usize;
            let entry = buf
                .tool_calls
                .entry(index)
                .or_insert_with(|| (id.clone(), name.clone(), String::new()));
            entry.0 = id;
            if name.is_some() {
                entry.1 = name;
            }
            entry.2.push_str(&args);
        }
        _ => {}
    }
}

/// Flush a step's buffered chunk deltas as transcript messages. Only called
/// for steps whose `assistant/message` never assembled (interrupted stream);
/// the normal path removes the buffer when the message arrives.
fn flush_step_chunks(state: &mut ParseState, turn: u32, step: u32) {
    let Some(buf) = state.chunk_bufs.remove(&(turn, step)) else {
        return;
    };
    let produced_start = state.messages.len();
    let mut indices: Vec<usize> = buf
        .text
        .keys()
        .chain(buf.reasoning.keys())
        .chain(buf.tool_calls.keys())
        .copied()
        .collect();
    indices.sort_unstable();
    indices.dedup();
    for index in indices {
        if let Some(text) = buf
            .reasoning
            .get(&index)
            .filter(|text| !text.trim().is_empty())
        {
            state
                .messages
                .push(Message::system(format!("[thinking]\n{text}")));
        }
        if let Some(text) = buf.text.get(&index).filter(|text| !text.trim().is_empty()) {
            state.content_parts.push(text.clone());
            state.messages.push(Message::assistant(text.clone()));
        }
        if let Some((call_id, name, arguments)) = buf.tool_calls.get(&index)
            && !state.tool_by_call_id.contains_key(call_id)
        {
            let name = name.as_deref().unwrap_or("tool");
            push_tool_message(state, name, arguments, Some(call_id), None);
        }
    }
    // Mirror the assembled-message path: the step's streamed usage lands in
    // `usage_events` and on the flush's last non-System message, so an
    // interrupted step still counts toward the session's tokens.
    if let Some(usage) = buf.usage {
        state.push_usage_event(&usage, None, buf.usage_timestamp.as_deref());
        if let Some(last) = state.messages[produced_start..]
            .iter_mut()
            .filter(|message| message.role != MessageRole::System)
            .last()
        {
            last.token_usage = Some(usage);
        } else {
            state.messages.push(Message {
                token_usage: Some(usage),
                ..Message::assistant(String::new())
            });
        }
    }
}

/// `$DSH_HOME/attachments/v1/objects` for a log at
/// `$DSH_HOME/sessions/<project-key>/<session-id>/<artifact>`; `None` for a
/// log outside that layout.
fn attachment_objects_dir(path: &Path) -> Option<PathBuf> {
    let sessions = path.ancestors().nth(3)?;
    if sessions.file_name()? != "sessions" {
        return None;
    }
    Some(
        sessions
            .parent()?
            .join("attachments")
            .join("v1")
            .join("objects"),
    )
}

/// Parse one DSH session artifact into a [`ParsedSession`]. Returns `None`
/// for files that cannot be opened/decompressed, lack a session header, or
/// contain no surfaced messages (a session that never started).
pub fn parse_session_file(path: &Path) -> Option<ParsedSession> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) => {
            log::warn!("failed to open DSH session '{}': {error}", path.display());
            return None;
        }
    };
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) => {
            log::warn!(
                "failed to read DSH session metadata '{}': {error}",
                path.display()
            );
            return None;
        }
    };
    let file_size = metadata.len();
    let source_mtime = metadata
        .modified()
        .ok()
        .and_then(system_time_to_epoch_seconds)
        .unwrap_or(0);
    let reader = match open_log_reader(path, file) {
        Ok(reader) => reader,
        Err(error) => {
            log::warn!("failed to open DSH session '{}': {error}", path.display());
            return None;
        }
    };
    let mut state = ParseState {
        attachment_objects: attachment_objects_dir(path),
        ..ParseState::default()
    };
    let header = scan_records(reader, path, &mut state)?;
    if state.seed_cut.take().is_some() {
        // The log ends inside its inherited prefix: nothing is local yet.
        state.drop_inherited_usage();
    }
    // Torn logs may lack the closing step/end; flush whatever is left.
    let mut pending_steps: Vec<(u32, u32)> = state.chunk_bufs.keys().copied().collect();
    // Deterministic order: flush in (turn, step) sequence, not HashMap order.
    pending_steps.sort_unstable();
    for (turn, step) in pending_steps {
        flush_step_chunks(&mut state, turn, step);
    }
    if state.messages.is_empty() {
        log::debug!(
            "skipping DSH session '{}': no surfaced messages",
            path.display()
        );
        return None;
    }
    let content_text = state.content_parts.join("\n");
    let meta = assemble_session_meta(path, &header, &state, file_size, source_mtime);
    Some(ParsedSession {
        meta,
        messages: state.messages,
        content_text,
        parse_warning_count: state.parse_warning_count,
        child_session_ids: Vec::new(),
        usage_events: state.usage_events,
        source_mtime,
    })
}

fn assemble_session_meta(
    path: &Path,
    header: &DshHeader,
    state: &ParseState,
    file_size: u64,
    source_mtime: i64,
) -> SessionMeta {
    let id = header.id.clone().unwrap_or_else(|| {
        path.parent()
            .and_then(|parent| parent.file_name())
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default()
    });
    let project_path = header.cwd.clone().unwrap_or_default();
    let project_name = Path::new(&project_path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    let created_at = header.created_at.unwrap_or(0) / 1000;
    let updated_at = state
        .last_event_time_ms
        .map(|time| time / 1000)
        .unwrap_or(source_mtime);
    let parent_id = header.parent_session.clone().filter(|id| !id.is_empty());
    // `parentSession` alone is seed lineage — a forked/resumed branch carries
    // it too. Only `origin: "subagent"` marks a delegated child.
    let is_sidechain = header.origin.as_deref() == Some("subagent");
    // A delegated session is never auto-titled, so the parent-chosen
    // delegation label wins when present. DSH's `fallback` title is only the
    // prompt's leading words — sibling workflow agents share it — so the full
    // first prompt outranks it. `first_user_text` is set by `push_user` for
    // every surfaced direct prompt, so it is exactly the first User message's
    // text; no scan needed.
    let title = if is_sidechain {
        state.descriptor_label.clone()
    } else {
        None
    }
    .or_else(|| state.latest_title.clone())
    .or_else(|| {
        state
            .first_user_text
            .as_deref()
            .map(|text| session_title(Some(text)))
    })
    .or_else(|| state.fallback_title.clone())
    .unwrap_or_else(|| session_title(None));
    SessionMeta {
        id,
        provider: Provider::Dsh,
        title,
        project_path,
        project_name,
        created_at,
        updated_at,
        message_count: state.messages.len() as u32,
        file_size_bytes: file_size,
        source_path: path.to_string_lossy().to_string(),
        is_sidechain,
        variant_name: state
            .agent_preset
            .clone()
            .or_else(|| header.agent_preset.clone()),
        model: state.model.clone(),
        cc_version: None,
        git_branch: None,
        parent_id,
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MessageRole;
    use crate::provider::SessionProvider;
    use crate::providers::dsh::DshProvider;
    use tempfile::TempDir;

    const SESSION_ID: &str = "session-11111111-aaaa-4aaa-8aaa-aaaaaaaaaaaa";

    fn header_line(cwd: &str) -> String {
        format!(
            r#"{{"type":"session","version":0,"id":"{SESSION_ID}","createdAt":1786865077879,"cwd":"{cwd}","delegationDepth":0,"agentPreset":"standard"}}"#
        )
    }

    fn event_line(event_type: &str, seq: u64, time: i64, data: &str) -> String {
        format!(r#"{{"type":"{event_type}","seq":{seq},"time":{time},"data":{data}}}"#)
    }

    fn user_message_line(seq: u64, time: i64, source: &str, text: &str) -> String {
        event_line(
            "user/message",
            seq,
            time,
            &format!(
                r#"{{"content":[{{"type":"text","text":{text}}}],"source":{source},"role":"user","id":"u{seq}"}}"#
            ),
        )
    }

    fn assistant_message_line(seq: u64, time: i64, content: &str, usage: Option<&str>) -> String {
        let usage_json = usage.map_or(String::new(), |u| format!(r#","usage":{u}"#));
        event_line(
            "assistant/message",
            seq,
            time,
            &format!(
                r#"{{"turn":1,"step":1,"message":{{"role":"assistant","content":{content},"source":{{"kind":"model","provider":"opencode-go","model":"deepseek-v4-flash"}},"id":"a{seq}"}}{usage_json}}}"#
            ),
        )
    }

    /// `arguments` is a JSON string literal (quotes included) — DSH stores the
    /// tool arguments as an escaped string, not an object.
    fn tool_call_line(seq: u64, time: i64, call_id: &str, name: &str, arguments: &str) -> String {
        event_line(
            "tool/call",
            seq,
            time,
            &format!(
                r#"{{"turn":1,"step":1,"callId":"{call_id}","name":"{name}","arguments":{arguments}}}"#
            ),
        )
    }

    fn tool_result_line(seq: u64, time: i64, call_id: &str, text: &str, is_error: bool) -> String {
        event_line(
            "tool/result",
            seq,
            time,
            &format!(
                r#"{{"turn":1,"step":1,"message":{{"source":{{"kind":"tool","callId":"{call_id}"}},"content":[{{"type":"tool-result","toolCallId":"{call_id}","content":[{{"type":"text","text":{text}}}],"isError":{is_error}}}],"role":"user","id":"r{seq}"}}}}"#
            ),
        )
    }

    /// A compaction checkpoint: a plugin `user/message` whose
    /// `surfaceOp: replace` shadows every surface event whose seq appears in
    /// `sourceEventSeqs`. `text` is a JSON string literal (quotes included).
    fn checkpoint_line(seq: u64, time: i64, text: &str, shadowed_seqs: &[u64]) -> String {
        let seqs = shadowed_seqs
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"type":"user/message","seq":{seq},"time":{time},"surfaceOp":{{"op":"replace","start":1,"end":3}},"sourceEventSeqs":[{seqs}],"data":{{"content":[{{"type":"text","text":{text}}}],"source":{{"kind":"plugin","plugin":"compact","compactionId":"c-{seq}","sourceCommandId":"cmd-{seq}"}},"role":"user","id":"cp-{seq}"}}}}"#
        )
    }

    fn write_log(dir: &TempDir, name: &str, lines: &[&str]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, lines.join("\n") + "\n").expect("fixture must be written");
        path
    }

    fn parse_lines(lines: &[&str]) -> ParsedSession {
        let dir = TempDir::new().expect("temp dir must be created");
        let path = write_log(&dir, "session.jsonl", lines);
        parse_session_file(&path).expect("fixture must parse")
    }

    /// A subagent header: `parentSession` + `origin` mark a delegated child.
    fn subagent_header_line(cwd: &str) -> String {
        format!(
            r#"{{"type":"session","version":0,"id":"{SESSION_ID}","createdAt":1786865077879,"cwd":"{cwd}","parentSession":"session-parent","origin":"subagent","delegationDepth":1,"agentPreset":"standard"}}"#
        )
    }

    #[test]
    fn subagent_title_prefers_descriptor_label() {
        let session = parse_lines(&[
            &subagent_header_line("/home/user/my-project"),
            &event_line(
                "subagent/descriptor",
                0,
                1786865077879,
                r#"{"version":2,"mode":"continuable","provider":"spawn","label":"Deep review all project code","agentProvider":"opencode-go","agentModel":"deepseek-v4-flash"}"#,
            ),
            &user_message_line(
                1,
                1786865077880,
                r#"{"kind":"user"}"#,
                r#""You are conducting a deep code review of…""#,
            ),
            &event_line(
                "session/title",
                2,
                1786865077881,
                r#"{"title":"You are conducting a deep","messageSeqs":[1],"source":{"kind":"fallback"}}"#,
            ),
        ]);
        assert!(session.meta.is_sidechain);
        assert_eq!(session.meta.parent_id.as_deref(), Some("session-parent"));
        assert_eq!(session.meta.title, "Deep review all project code");
    }

    #[test]
    fn subagent_report_renders_as_tagged_system_row() {
        let session = parse_lines(&[
            &header_line("/home/user/my-project"),
            &user_message_line(1, 1786865077880, r#"{"kind":"user"}"#, r#""hello""#),
            &event_line(
                "user/message",
                2,
                1786865077881,
                r#"{"content":[{"type":"text","text":"Background subagent cb61a6b7 reported:"},{"type":"text","text":"Review complete."}],"source":{"kind":"subagent-report","form":"relay","senderSessionId":"cb61a6b7"},"role":"user","id":"u2"}"#,
            ),
        ]);
        let report = session
            .messages
            .iter()
            .find(|message| message.content.starts_with("[subagent_report]"))
            .expect("report row");
        assert_eq!(report.role, MessageRole::System);
        assert_eq!(report.content, "[subagent_report] Review complete.");
    }

    #[test]
    fn subagent_settled_strips_wrapper_and_tags_closing_message() {
        let session = parse_lines(&[
            &header_line("/home/user/my-project"),
            &user_message_line(1, 1786865077880, r#"{"kind":"user"}"#, r#""hello""#),
            &event_line(
                "user/message",
                2,
                1786865077881,
                r#"{"content":[{"type":"text","text":"Background subagent 036bdb9c finished and will do no further work unless you send it more."},{"type":"text","text":"Its closing message:"},{"type":"text","text":"Review complete.\nAll good."}],"source":{"kind":"subagent-settled","form":"notice","senderSessionId":"036bdb9c"},"role":"user","id":"u2"}"#,
            ),
        ]);
        let settled = session
            .messages
            .iter()
            .find(|message| message.content.starts_with("[subagent_settled]"))
            .expect("settled row");
        assert_eq!(settled.role, MessageRole::System);
        assert_eq!(
            settled.content,
            "[subagent_settled] Review complete.\nAll good."
        );
        assert!(!settled.content.contains("Background subagent"));
    }

    #[test]
    fn forked_session_is_not_a_sidechain() {
        // `parentSession` without `origin: subagent` is seed lineage (a
        // fork/resume branch), not a delegated child.
        let session = parse_lines(&[
            &format!(
                r#"{{"type":"session","version":0,"id":"{SESSION_ID}","createdAt":1786865077879,"cwd":"/home/user/my-project","parentSession":"session-parent","seedLength":4}}"#
            ),
            &user_message_line(1, 1786865077880, r#"{"kind":"user"}"#, r#""hello""#),
        ]);
        assert!(!session.meta.is_sidechain);
        assert_eq!(session.meta.parent_id.as_deref(), Some("session-parent"));
    }

    #[test]
    fn interrupted_step_keeps_streamed_usage() {
        // A step whose assistant/message never assembled: text arrives as
        // chunks, usage as a usage chunk, then the stream dies. The flush
        // must keep both the text and the token accounting.
        let session = parse_lines(&[
            &header_line("/home/user/my-project"),
            &user_message_line(1, 1786865077880, r#"{"kind":"user"}"#, r#""hello""#),
            &assistant_message_line(
                2,
                1786865077881,
                r#"[{"type":"text","text":"first"}]"#,
                Some(r#"{"inputTokens":7,"outputTokens":3}"#),
            ),
            &event_line(
                "assistant/chunk",
                3,
                1786865077890,
                r#"{"turn":1,"step":2,"chunk":{"type":"text-delta","index":0,"text":"partial answer"}}"#,
            ),
            &event_line(
                "assistant/chunk",
                4,
                1786865077891,
                r#"{"turn":1,"step":2,"chunk":{"type":"usage","usage":{"inputTokens":100,"outputTokens":20,"cacheReadTokens":40}}}"#,
            ),
        ]);
        assert_eq!(session.usage_events.len(), 2);
        let orphan = &session.usage_events[1];
        assert_eq!(orphan.input_tokens, 100);
        assert_eq!(orphan.output_tokens, 20);
        assert_eq!(orphan.cache_read_input_tokens, 40);
        assert_eq!(orphan.model, "deepseek-v4-flash");
        let last = session.messages.last().expect("flushed message");
        let usage = last.token_usage.as_ref().expect("usage attached");
        assert_eq!(usage.input_tokens, 100);
    }

    /// Full fixture: user prompt, reasoning + text + tool call, tool result,
    /// step end, and a session title event.
    fn full_roundtrip_session() -> ParsedSession {
        let dir = TempDir::new().unwrap();
        let path = write_log(
            &dir,
            "session.jsonl",
            &[
                &header_line("/home/user/my-project"),
                &user_message_line(
                    1,
                    1786865127218,
                    r#"{"kind":"user","rpcId":"rpc-1"}"#,
                    r#""hello dsh""#,
                ),
                &assistant_message_line(
                    2,
                    1786865132303,
                    r#"[{"type":"reasoning","text":"let me think"},{"type":"text","text":"hi there"},{"type":"tool-call","id":"call_1","name":"bash","arguments":"{\"command\":\"ls\"}"}]"#,
                    Some(
                        r#"{"inputTokens":100,"outputTokens":25,"cacheReadTokens":50,"cacheWriteTokens":10}"#,
                    ),
                ),
                &tool_result_line(3, 1786865132305, "call_1", r#""file.txt""#, false),
                &event_line("step/end", 4, 1786865133000, r#"{"turn":1,"step":1}"#),
                &event_line(
                    "session/title",
                    5,
                    1786865134000,
                    r#"{"title":"A Great Title","messageSeqs":[1],"source":{"kind":"provider","provider":"session-title-first-prompt-llm"}}"#,
                ),
            ],
        );
        parse_session_file(&path).expect("fixture must parse")
    }

    #[test]
    fn parses_full_session_meta() {
        let session = full_roundtrip_session();
        assert_eq!(session.meta.id, SESSION_ID);
        assert_eq!(session.meta.title, "A Great Title");
        assert_eq!(session.meta.project_path, "/home/user/my-project");
        assert_eq!(session.meta.project_name, "my-project");
        assert_eq!(session.meta.created_at, 1786865077);
        assert_eq!(session.meta.updated_at, 1786865134);
        assert_eq!(session.meta.model.as_deref(), Some("deepseek-v4-flash"));
        assert_eq!(session.meta.parent_id, None);
        assert!(!session.meta.is_sidechain);
        assert_eq!(session.meta.message_count, 4);
        assert_eq!(session.parse_warning_count, 0);
        assert!(session.content_text.contains("hello dsh"));
        assert!(session.content_text.contains("hi there"));
        assert!(!session.content_text.contains("let me think"));
    }

    #[test]
    fn parses_full_session_messages() {
        let session = full_roundtrip_session();
        let messages = &session.messages;
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[0].content, "hello dsh");
        assert_eq!(messages[1].role, MessageRole::System);
        assert_eq!(messages[1].content, "[thinking]\nlet me think");
        assert_eq!(messages[2].role, MessageRole::Assistant);
        assert_eq!(messages[2].content, "hi there");
        assert_eq!(messages[2].model.as_deref(), Some("deepseek-v4-flash"));
        assert!(messages[2].token_usage.is_none());
        assert_eq!(messages[3].role, MessageRole::Tool);
        assert_eq!(messages[3].tool_name.as_deref(), Some("Bash"));
        // Usage attaches to the event's last non-System message (the tool
        // message), mirroring the Claude provider's convention.
        let usage = messages[3].token_usage.as_ref().expect("usage attached");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 25);
        assert_eq!(usage.cache_read_input_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, 10);
        assert_eq!(messages[3].content, "file.txt");
        let metadata = messages[3].tool_metadata.as_ref().unwrap();
        assert_eq!(metadata.status.as_deref(), Some("success"));
    }

    #[test]
    fn title_falls_back_to_first_user_message() {
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &user_message_line(1, 1, r#"{"kind":"user"}"#, r#""hello dsh""#),
        ]);
        assert_eq!(session.meta.title, "hello dsh");
    }

    #[test]
    fn maps_tool_result_error_status() {
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &assistant_message_line(
                1,
                1786865132303,
                r#"[{"type":"tool-call","id":"call_9","name":"glob","arguments":"{}"}]"#,
                None,
            ),
            &tool_result_line(2, 1786865132305, "call_9", r#""boom""#, true),
        ]);
        let tool = session.messages.last().unwrap();
        assert_eq!(tool.role, MessageRole::Tool);
        assert_eq!(tool.content, "boom");
        assert_eq!(
            tool.tool_metadata.as_ref().unwrap().status.as_deref(),
            Some("error")
        );
    }

    #[test]
    fn drops_agent_instructions_and_snapshot_noise() {
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &user_message_line(1, 1, r#"{"kind":"user"}"#, r#""real prompt""#),
            &user_message_line(
                2,
                2,
                r#"{"kind":"agent-instructions","form":"instructions"}"#,
                r#""<system-reminder>AGENTS.md</system-reminder>""#,
            ),
            &user_message_line(
                3,
                3,
                r#"{"kind":"plugin","plugin":"@deepseek-ai/dsh-system-prompt","form":"snapshot"}"#,
                r#""runtime snapshot""#,
            ),
            &user_message_line(
                4,
                4,
                r#"{"kind":"plugin","plugin":"dsh-tool-skill","form":"catalog"}"#,
                r#""skill catalog dump""#,
            ),
            &user_message_line(
                5,
                5,
                r#"{"kind":"plugin","plugin":"dsh-session-reference","form":"recall"}"#,
                r#""recalled context from another session""#,
            ),
            &user_message_line(
                6,
                6,
                r#"{"kind":"plugin","plugin":"user-approval"}"#,
                r#""The approval policy changed from ask to never""#,
            ),
        ]);
        let messages = &session.messages;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[0].content, "real prompt");
        assert_eq!(messages[1].role, MessageRole::System);
        assert!(messages[1].content.contains("approval policy"));
    }

    #[test]
    fn plain_string_content_is_kept() {
        // Older/foreign producers may emit `content` as a bare string.
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &event_line(
                "user/message",
                1,
                1,
                r#"{"content":"plain string prompt","source":{"kind":"user"},"role":"user","id":"u1"}"#,
            ),
        ]);
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].role, MessageRole::User);
        assert_eq!(session.messages[0].content, "plain string prompt");
    }

    #[test]
    fn interrupts_do_not_lose_streamed_text() {
        // A step whose assistant/message never assembled: chunks only.
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &event_line(
                "assistant/chunk",
                1,
                1000,
                r#"{"turn":1,"step":1,"chunk":{"type":"block-start","index":0,"blockType":"reasoning"}}"#,
            ),
            &event_line(
                "reasoning-chunks",
                2,
                1000,
                r#"{"turn":1,"step":1,"index":0,"dt":[1],"texts":["why not"]}"#,
            ),
            &event_line(
                "text-chunks",
                3,
                1000,
                r#"{"turn":1,"step":1,"index":1,"dt":[1],"texts":["partial ","answer"]}"#,
            ),
            &event_line(
                "tool-call-chunks",
                4,
                1000,
                r#"{"turn":1,"step":1,"index":2,"dt":[1],"id":"call_7","name":"bash","args":["{","}"]}"#,
            ),
            &event_line("step/end", 5, 2000, r#"{"turn":1,"step":1}"#),
            &tool_result_line(6, 2001, "call_7", r#""out""#, false),
        ]);
        let messages = &session.messages;
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[0].content, "[thinking]\nwhy not");
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(messages[1].content, "partial answer");
        assert_eq!(messages[2].role, MessageRole::Tool);
        assert_eq!(messages[2].content, "out");
        assert_eq!(messages[2].tool_name.as_deref(), Some("Bash"));
    }

    #[test]
    fn late_chunk_rows_do_not_duplicate_assembled_text() {
        // The step's assistant/message arrived first, then an out-of-order
        // chunk row follows; the chunk must not re-buffer (and later flush)
        // a duplicate of the assembled text.
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &assistant_message_line(
                1,
                1000,
                r#"[{"type":"text","text":"assembled answer"}]"#,
                None,
            ),
            &event_line(
                "text-chunks",
                2,
                1001,
                r#"{"turn":1,"step":1,"index":0,"dt":[1],"texts":["assembled answer"]}"#,
            ),
            &event_line("step/end", 3, 2000, r#"{"turn":1,"step":1}"#),
        ]);
        let messages = &session.messages;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "assembled answer");
    }

    #[test]
    fn compaction_keeps_the_shadowed_history_and_marks_the_checkpoint() {
        // Real compaction shape: bracket rows, then a checkpoint user/message
        // carrying `surfaceOp: {op: "replace"}` whose sourceEventSeqs cite
        // every shadowed surface event. The originals stay; the checkpoint
        // shows its summary under the compaction marker.
        let framed =
            r#""Checkpoint preamble.\n\n<compacted-summary>condensed history</compacted-summary>""#;
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""old prompt""#),
            &assistant_message_line(
                2,
                1001,
                r#"[{"type":"text","text":"old answer"},{"type":"tool-call","id":"call_1","name":"bash","arguments":"{}"}]"#,
                Some(r#"{"inputTokens":300,"outputTokens":50}"#),
            ),
            &tool_result_line(3, 1002, "call_1", r#""old output""#, false),
            &event_line(
                "compaction/start",
                4,
                2000,
                r#"{"compactionId":"c-1","sourceCommandId":"cmd-1","turn":null}"#,
            ),
            &event_line(
                "compaction/summary",
                5,
                2001,
                r#"{"compactionId":"c-1","sourceCommandId":"cmd-1","summary":[{"type":"text","text":"condensed history"}]}"#,
            ),
            &checkpoint_line(6, 2002, framed, &[1, 2, 3]),
            &event_line(
                "compaction/end",
                7,
                2003,
                r#"{"compactionId":"c-1","sourceCommandId":"cmd-1","turn":null}"#,
            ),
            &user_message_line(
                8,
                2004,
                r#"{"kind":"user"}"#,
                r#""new prompt after compaction""#,
            ),
        ]);
        let contents: Vec<&str> = session
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            contents,
            [
                "old prompt",
                "old answer",
                "old output",
                "[context_compacted]\ncondensed history",
                "new prompt after compaction",
            ]
        );
        assert_eq!(session.messages[3].role, MessageRole::System);
        assert_eq!(session.usage_events.len(), 1);
        assert_eq!(session.usage_events[0].input_tokens, 300);
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn v4_checkpoints_use_the_compact_checkpoint_source() {
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""old prompt""#),
            r#"{"type":"user/message","seq":2,"time":2000,"surfaceOp":{"op":"replace","startSeq":1,"endSeq":1},"sourceEventSeqs":[1],"data":{"content":[{"type":"text","text":"Preamble.\n\n<compacted-summary>"},{"type":"text","text":"the gist"},{"type":"text","text":"</compacted-summary>"}],"source":{"kind":"compact-checkpoint","compactionId":"c-1"},"role":"user","id":"cp-2"}}"#,
        ]);
        let contents: Vec<&str> = session
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(contents, ["old prompt", "[context_compacted]\nthe gist"]);
    }

    #[test]
    fn usage_event_model_falls_back_to_session_model() {
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &assistant_message_line(1, 1000, r#"[{"type":"text","text":"first"}]"#, None),
            // Second event reports usage but its source carries no model.
            &event_line(
                "assistant/message",
                2,
                1001,
                r#"{"turn":1,"step":2,"message":{"role":"assistant","content":[{"type":"text","text":"second"}],"source":{"kind":"model"},"id":"a2"},"usage":{"inputTokens":10,"outputTokens":5}}"#,
            ),
        ]);
        assert_eq!(session.usage_events.len(), 1);
        assert_eq!(session.usage_events[0].model, "deepseek-v4-flash");
        assert_eq!(session.usage_events[0].input_tokens, 10);
        assert_eq!(session.usage_events[0].output_tokens, 5);
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn usage_event_without_any_model_is_a_counted_warning() {
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &event_line(
                "assistant/message",
                1,
                1000,
                r#"{"turn":1,"step":1,"message":{"role":"assistant","content":[{"type":"text","text":"answer"}],"source":{"kind":"model"},"id":"a1"},"usage":{"inputTokens":10,"outputTokens":5}}"#,
            ),
        ]);
        assert!(session.usage_events.is_empty());
        assert_eq!(session.parse_warning_count, 1);
        // The message still carries its usage for display.
        let usage = session.messages[0].token_usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, 10);
    }

    #[test]
    fn retry_and_subagent_metadata_events_are_not_warnings() {
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &event_line(
                "llm/retry",
                1,
                1000,
                r#"{"retryId":"r-1","turn":1,"step":1,"provider":"opencode-go","mode":"normal","retry":1,"maxRetries":2,"delayMs":512.4,"failure":{"message":"boom","code":"TRANSPORT"}}"#,
            ),
            &event_line(
                "llm/retry-started",
                2,
                1001,
                r#"{"retryId":"r-1","turn":1,"step":1,"retry":1}"#,
            ),
            &event_line(
                "subagent/descriptor",
                3,
                1002,
                r#"{"version":2,"mode":"one-shot","provider":"spawn","label":"check code"}"#,
            ),
            &user_message_line(4, 1003, r#"{"kind":"user"}"#, r#""still works""#),
        ]);
        let contents: Vec<&str> = session
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            contents,
            [
                "[retry] retry 1/2 after 512 ms (TRANSPORT)\nboom",
                "still works"
            ]
        );
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn tolerates_torn_tail_and_malformed_lines() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("session.jsonl");
        let body = format!(
            "{}\n{}\n{}\n{}",
            header_line("/tmp/p"),
            user_message_line(1, 1, r#"{"kind":"user"}"#, r#""ok""#),
            r#"{"type":"user/message","seq":2,"time":2,"data":{"content":[{"type":"text","text":"broken"}]"#,
            r#"{"type":"totally-unknown","seq":3,"time":3,"data":{},"ignorable":true}"#,
        );
        // No trailing newline: the final record is a torn tail.
        std::fs::write(&path, body).unwrap();

        let session = parse_session_file(&path).expect("fixture must parse");
        assert_eq!(session.messages.len(), 1);
        // Malformed line + missing newline: malformed counts, torn tail doesn't.
        assert_eq!(session.parse_warning_count, 1);
    }

    #[test]
    fn unknown_required_event_counts_as_parse_warning() {
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &user_message_line(1, 1, r#"{"kind":"user"}"#, r#""ok""#),
            &event_line("mystery/event", 2, 2, r#"{"x":1}"#),
        ]);
        assert_eq!(session.parse_warning_count, 1);
    }

    #[test]
    fn current_metadata_and_log_only_events_are_handled() {
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &user_message_line(1, 1, r#"{"kind":"user"}"#, r#""ok""#),
            &event_line(
                "web/deepseek-search-llm-request",
                2,
                2,
                r#"{"model":"deepseek-search"}"#,
            ),
            &event_line(
                "agent-preset/selected",
                3,
                3,
                r#"{"agentPreset":"reviewer"}"#,
            ),
        ]);

        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.meta.variant_name.as_deref(), Some("reviewer"));
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn malformed_agent_preset_selection_remains_a_parse_warning() {
        let session = parse_lines(&[
            &header_line("/tmp/p"),
            &user_message_line(1, 1, r#"{"kind":"user"}"#, r#""ok""#),
            &event_line("agent-preset/selected", 2, 2, r#"{}"#),
        ]);

        assert_eq!(session.meta.variant_name.as_deref(), Some("standard"));
        assert_eq!(session.parse_warning_count, 1);
    }

    #[test]
    fn reads_zstd_compressed_artifacts() {
        let dir = TempDir::new().unwrap();
        let plain = format!(
            "{}\n{}\n",
            header_line("/tmp/p"),
            user_message_line(1, 1, r#"{"kind":"user"}"#, r#""from zstd""#),
        );
        let path = dir.path().join("session.jsonl.zstd");
        let compressed = zstd::stream::encode_all(plain.as_bytes(), 3).unwrap();
        std::fs::write(&path, compressed).unwrap();

        let session = parse_session_file(&path).expect("zstd fixture must parse");
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].content, "from zstd");
        assert_eq!(session.meta.source_path, path.to_string_lossy());
    }

    #[test]
    fn provider_scans_fake_home_and_loads_messages() {
        let dir = TempDir::new().unwrap();
        let sessions = dir.path().join("sessions");
        let project = sessions.join("--tmp-p--");
        let session_dir = project.join(SESSION_ID);
        std::fs::create_dir_all(&session_dir).unwrap();
        let log = format!(
            "{}\n{}\n",
            header_line("/tmp/p"),
            user_message_line(1, 1, r#"{"kind":"user"}"#, r#""scan me""#),
        );
        std::fs::write(session_dir.join("session.jsonl"), log).unwrap();

        let provider = DshProvider::with_home(dir.path().to_path_buf());
        assert_eq!(provider.provider(), Provider::Dsh);

        let sessions = provider.scan_all().expect("scan must succeed");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].meta.id, SESSION_ID);
        assert_eq!(sessions[0].meta.project_path, "/tmp/p");

        let loaded = provider
            .load_messages(SESSION_ID, &sessions[0].meta.source_path)
            .expect("load must succeed");
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].role, MessageRole::User);

        // Incremental scan short-circuits the unchanged file.
        use std::collections::HashMap;
        let known: HashMap<String, crate::provider::SourceState> = provider
            .scan_all()
            .unwrap()
            .into_iter()
            .map(|s| {
                (
                    s.meta.source_path.clone(),
                    crate::provider::SourceState {
                        size: s.meta.file_size_bytes,
                        mtime: s.source_mtime,
                        title: None,
                    },
                )
            })
            .collect();
        let outcome = provider.scan_incremental(&known).expect("incremental scan");
        assert!(outcome.parsed.is_empty());
        assert_eq!(outcome.unchanged_source_paths.len(), 1);
    }

    /// A v4 header: the framing v2+ generations share with v0, plus the
    /// fields newer releases add (`isSeeded`, `delegationDepth`).
    fn versioned_header_line(version: i64, cwd: &str) -> String {
        format!(
            r#"{{"type":"session","version":{version},"id":"{SESSION_ID}","createdAt":1786865077879,"cwd":"{cwd}","isSeeded":false,"delegationDepth":0,"agentPreset":"standard"}}"#
        )
    }

    #[test]
    fn versioned_headers_parse_without_warnings() {
        for version in [2, 3, 4] {
            let session = parse_lines(&[
                &versioned_header_line(version, "/tmp/p"),
                &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""hello v4""#),
            ]);
            assert_eq!(session.meta.id, SESSION_ID);
            assert_eq!(session.messages.len(), 1);
            assert_eq!(
                session.parse_warning_count, 0,
                "v{version} header must not warn"
            );
        }
    }

    #[test]
    fn unknown_future_header_version_warns() {
        let session = parse_lines(&[
            &versioned_header_line(99, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""hello future""#),
        ]);
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.parse_warning_count, 1);
    }

    #[test]
    fn v2_generation_rows_are_silent() {
        // Every new row type the v2+ generation adds that carries no
        // transcript semantics must parse warning-free.
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""hello""#),
            &event_line(
                "assistant/attempt",
                2,
                1001,
                r#"{"turn":1,"step":1,"stream":[{"type":"chunk","time":1001,"chunk":{"type":"usage","usage":{"inputTokens":0,"outputTokens":0,"totalTokens":0}}},{"type":"chunk","time":1001,"chunk":{"type":"finish","reason":{"kind":"error"}}}]}"#,
            ),
            &event_line("workspace/changes", 5, 1004, r#"{"turn":1}"#),
            &event_line(
                "model/selection",
                6,
                1005,
                r#"{"provider":"aittest","model":"dsv41"}"#,
            ),
            &event_line(
                "session-log-deepseek/delivery-accepted",
                7,
                1006,
                r#"{"sessionId":"session-x","sessionFormatVersion":4,"throughSeq":6}"#,
            ),
            &event_line(
                "approval/asked",
                8,
                1007,
                r#"{"id":"a-1","toolName":"bash","callId":"call_1","reason":"escalate"}"#,
            ),
            &event_line(
                "approval/decided",
                9,
                1008,
                r#"{"id":"a-1","outcome":"allowed-once"}"#,
            ),
            &event_line("activity/status", 10, 1009, r#"{"status":"running"}"#),
        ]);
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].content, "hello");
        // A failed attempt that reported zero tokens bills nothing.
        assert!(session.usage_events.is_empty());
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn system_message_prompt_dump_is_dropped() {
        // `system/message` carries the session prompt with surfaceOp append;
        // it is context, not conversation — a prompt-only log yields nothing.
        let dir = TempDir::new().expect("temp dir must be created");
        let path = write_log(
            &dir,
            "session.jsonl",
            &[
                &versioned_header_line(4, "/tmp/p"),
                r#"{"type":"system/message","seq":1,"time":1000,"surfaceOp":"append","data":{"turn":1,"step":1,"message":{"role":"system","content":[{"type":"text","text":"You are a helpful software engineer assistant."}],"source":{"kind":"plugin","plugin":"@deepseek-ai/dsh-system-prompt"},"id":"s1"}}}"#,
            ],
        );
        assert!(parse_session_file(&path).is_none());
    }

    #[test]
    fn deliverables_surface_as_tagged_system_row() {
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""write a guide""#),
            &event_line(
                "deliverables/presented",
                2,
                1001,
                r#"{"turn":1,"callId":"call_1","files":[{"path":"/tmp/guide.md","description":"Workspace guide"}]}"#,
            ),
        ]);
        assert_eq!(session.messages.len(), 2);
        let deliverables = &session.messages[1];
        assert_eq!(deliverables.role, MessageRole::System);
        assert_eq!(
            deliverables.content,
            "[deliverables]\n/tmp/guide.md — Workspace guide"
        );
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn drops_new_generation_context_dumps_and_keeps_notices() {
        let session = parse_lines(&[
            &versioned_header_line(3, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""hello""#),
            &event_line(
                "user/message",
                2,
                1001,
                r#"{"content":[{"type":"text","text":"snapshot"}],"source":{"kind":"runtime-context","form":"snapshot"},"role":"user","id":"u2"}"#,
            ),
            &event_line(
                "user/message",
                3,
                1002,
                r#"{"content":[{"type":"text","text":"catalog"}],"source":{"kind":"skill-catalog","form":"catalog"},"role":"user","id":"u3"}"#,
            ),
            &event_line(
                "user/message",
                4,
                1003,
                r#"{"content":[{"type":"text","text":"The approval policy changed."}],"source":{"kind":"user-approval"},"role":"user","id":"u4"}"#,
            ),
            // A coordinator relay is an injected instruction, not a dump: it
            // shares `form: "relay"` with subagent reports and must survive.
            &event_line(
                "user/message",
                5,
                1004,
                r#"{"content":[{"type":"text","text":"Resume the watch now."}],"source":{"kind":"coordinator","form":"relay","senderSessionId":"session-x"},"role":"user","id":"u5"}"#,
            ),
        ]);
        // Dumps gone; the approval notice and the relay render as System lines.
        assert_eq!(session.messages.len(), 3);
        assert_eq!(session.messages[0].content, "hello");
        assert_eq!(session.messages[1].role, MessageRole::System);
        assert_eq!(session.messages[1].content, "The approval policy changed.");
        assert_eq!(session.messages[2].role, MessageRole::System);
        assert_eq!(session.messages[2].content, "Resume the watch now.");
        assert_eq!(session.parse_warning_count, 0);
    }

    /// A v4 `tool/result`: the result blocks sit directly on a
    /// `role: "tool"` message. `envelope` carries extra envelope members
    /// (e.g. a replacement's `surfaceOp`), each with a leading comma.
    fn v4_tool_result_line(
        seq: u64,
        call_id: &str,
        text: &str,
        is_error: bool,
        envelope: &str,
    ) -> String {
        let time = 1000 + seq;
        format!(
            r#"{{"type":"tool/result","seq":{seq},"time":{time}{envelope},"data":{{"turn":1,"step":1,"message":{{"id":"r{seq}","role":"tool","toolCallId":"{call_id}","isError":{is_error},"source":{{"kind":"tool","callId":"{call_id}"}},"content":[{{"type":"text","text":{text}}}]}}}}}}"#
        )
    }

    fn tool_messages(session: &ParsedSession) -> Vec<&Message> {
        session
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::Tool)
            .collect()
    }

    #[test]
    fn v4_tool_results_attach_their_output_and_error_flag() {
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""run both""#),
            &tool_call_line(2, 1001, "call-ok", "bash", r#""{\"command\":\"ls\"}""#),
            &tool_call_line(3, 1002, "call-bad", "bash", r#""{\"command\":\"nope\"}""#),
            &v4_tool_result_line(4, "call-ok", r#""listing""#, false, ""),
            &v4_tool_result_line(5, "call-bad", r#""nope: not found""#, true, ""),
        ]);
        let tools = tool_messages(&session);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].content, "listing");
        assert_eq!(tools[1].content, "nope: not found");
        let status = |message: &Message| {
            message
                .tool_metadata
                .as_ref()
                .and_then(|metadata| metadata.status.clone())
        };
        assert_eq!(status(tools[0]).as_deref(), Some("success"));
        assert_eq!(status(tools[1]).as_deref(), Some("error"));
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn trimmed_tool_result_replacements_keep_the_full_output() {
        // DSH's tool-result pruner logs a `compaction/prune` shadow price and
        // then a replacement result trimmed for the model. The transcript
        // keeps the full output, for a surfaced call and an orphan alike.
        let trimmed = |seq: u64, call_id: &str, shadowed: u64| {
            v4_tool_result_line(
                seq,
                call_id,
                r#""full…[pruned]""#,
                false,
                &format!(
                    r#","surfaceOp":{{"op":"replace","startSeq":{shadowed},"endSeq":{shadowed}}},"sourceEventSeqs":[{shadowed}]"#
                ),
            )
        };
        let prune = |seq: u64, shadowed: u64| {
            event_line(
                "compaction/prune",
                seq,
                1006,
                &format!(
                    r#"{{"shadowedRange":{{"start":{shadowed},"end":{shadowed}}},"shadowedSeqs":[{shadowed}],"shadowedTokenCount":42}}"#
                ),
            )
        };
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""go""#),
            &tool_call_line(2, 1002, "call-1", "bash", r#""{\"command\":\"ls\"}""#),
            &v4_tool_result_line(3, "call-1", r#""full output""#, false, ""),
            &v4_tool_result_line(4, "orphan-1", r#""orphan output""#, false, ""),
            &assistant_message_line(5, 1005, r#"[{"type":"text","text":"done"}]"#, None),
            &prune(6, 3),
            &trimmed(7, "call-1", 3),
            &prune(8, 4),
            &trimmed(9, "orphan-1", 4),
        ]);
        let contents: Vec<&str> = session
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(contents, ["go", "full output", "orphan output", "done"]);
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn developer_messages_are_dropped_as_context() {
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""hello""#),
            r#"{"type":"developer/message","seq":2,"time":1001,"surfaceOp":"append","data":{"turn":1,"step":1,"message":{"id":"d2","role":"developer","source":{"kind":"tool-cordis"},"content":[{"type":"tool-addition","toolName":"bash"}]},"headerSeq":0}}"#,
        ]);
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn seeded_sessions_count_only_their_own_usage() {
        let usage = Some(r#"{"inputTokens":100,"outputTokens":10}"#);
        let answer = r#"[{"type":"text","text":"answer"}]"#;
        let fork_header = format!(
            r#"{{"type":"session","version":4,"id":"{SESSION_ID}","createdAt":1786865077879,"cwd":"/tmp/p","isSeeded":true,"parentSession":"session-parent","delegationDepth":0}}"#
        );
        let v0_header = |seed_length: u64| {
            format!(
                r#"{{"type":"session","version":0,"id":"{SESSION_ID}","createdAt":1786865077879,"cwd":"/tmp/p","parentSession":"session-parent","seedLength":{seed_length}}}"#
            )
        };
        let inherited = [
            user_message_line(0, 1000, r#"{"kind":"user"}"#, r#""parent prompt""#),
            assistant_message_line(1, 1001, answer, usage),
        ];
        let local = [
            user_message_line(3, 2000, r#"{"kind":"user"}"#, r#""fork prompt""#),
            assistant_message_line(4, 2001, answer, usage),
        ];
        let end_seed = |tag: &str| event_line("session/end-seed", 2, 1002, tag);

        let cases = [
            // v2+: the last marker tagged `inherited` ends the copied prefix.
            (
                fork_header,
                Some(end_seed(r#"{"inherited":true}"#)),
                true,
                1,
            ),
            // An unseeded session's untagged marker carries no cut.
            (
                versioned_header_line(4, "/tmp/p"),
                Some(end_seed("{}")),
                true,
                2,
            ),
            // v0/v1: the header's `seedLength` counts the copied events.
            (v0_header(2), None, true, 1),
            // A fork with no local events yet inherits everything.
            (v0_header(2), None, false, 0),
        ];
        for (header, marker, with_local, local_usage) in cases {
            let mut lines: Vec<&str> = vec![&header];
            lines.extend(inherited.iter().map(String::as_str));
            lines.extend(marker.as_deref());
            if with_local {
                lines.extend(local.iter().map(String::as_str));
            }
            let session = parse_lines(&lines);
            assert_eq!(session.usage_events.len(), local_usage, "{header}");
            let message_usage = session
                .messages
                .iter()
                .filter(|m| m.token_usage.is_some())
                .count();
            assert_eq!(message_usage, local_usage, "{header}");
            // The inherited prefix still renders.
            assert_eq!(session.messages[0].content, "parent prompt");
        }
    }

    /// Two zstd frames — `first` lines, then `second` lines — the way DSH
    /// appends one checksummed frame per durable batch.
    fn two_frame_log(first: &[&str], second: &[&str]) -> (Vec<u8>, usize) {
        let frame = |lines: &[&str]| {
            zstd::stream::encode_all((lines.join("\n") + "\n").as_bytes(), 3).unwrap()
        };
        let mut bytes = frame(first);
        let first_len = bytes.len();
        bytes.extend(frame(second));
        (bytes, first_len)
    }

    #[test]
    fn torn_final_zstd_frame_keeps_the_complete_records() {
        let header = versioned_header_line(4, "/tmp/p");
        let prompt = user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""kept""#);
        let lost = user_message_line(2, 1001, r#"{"kind":"user"}"#, r#""lost""#);
        let (bytes, first_len) = two_frame_log(&[&header, &prompt], &[&lost]);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("session.v4.jsonl.zstd");
        // Cut the second frame mid-write.
        std::fs::write(&path, &bytes[..first_len + (bytes.len() - first_len) / 2]).unwrap();

        let session = parse_session_file(&path).expect("torn log must still parse");
        let contents: Vec<&str> = session
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(contents, ["kept"]);
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn corrupt_zstd_data_keeps_the_readable_prefix_as_a_warning() {
        let header = versioned_header_line(4, "/tmp/p");
        let prompt = user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""kept""#);
        let (mut bytes, first_len) = two_frame_log(&[&header, &prompt], &[&prompt]);
        bytes.truncate(first_len);
        bytes.extend_from_slice(b"definitely not a zstd frame, but complete bytes");
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("session.v4.jsonl.zstd");
        std::fs::write(&path, &bytes).unwrap();

        let session = parse_session_file(&path).expect("prefix must still parse");
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.parse_warning_count, 1);
    }

    /// DSH 0.2's complete event vocabulary (`KNOWN_SESSION_EVENT_TYPES` in
    /// `@deepseek-ai/dsh-session`). None of these may count as unknown.
    const KNOWN_DSH_EVENT_TYPES: &[&str] = &[
        "agent-preset/selected",
        "agent/inbox/spliced",
        "approval/asked",
        "approval/decided",
        "approval/policy",
        "assistant/attempt",
        "assistant/message",
        "command/done",
        "command/run",
        "compaction/end",
        "compaction/prune",
        "compaction/start",
        "compaction/summary",
        "deliverables/presented",
        "developer/message",
        "feedback/message-delete",
        "feedback/message-put",
        "feedback/record",
        "goal/change",
        "hook/invoked",
        "hook/result",
        "image/offload",
        "llm/retry",
        "llm/retry-started",
        "model/selection",
        "permission/preset",
        "plan/mode",
        "request/context",
        "request/header",
        "sandbox/mode",
        "schedule/change",
        "session-log-deepseek/delivery-accepted",
        "session/end-seed",
        "session/title",
        "session/title-llm-request",
        "step/end",
        "step/start",
        "subagent/catalog",
        "subagent/descriptor",
        "subagent/model-selection-policy",
        "system/message",
        "team/member",
        "team/message/delivered",
        "team/message/queued",
        "team/task",
        "todo/write",
        "tool-workflow/agent-end",
        "tool-workflow/agent-start",
        "tool-workflow/run-end",
        "tool-workflow/run-start",
        "tool/call",
        "tool/ptc-dispatch",
        "tool/ptc-dispatch-start",
        "tool/result",
        "turn/end",
        "turn/start",
        "user/message",
        "web/deepseek-search-llm-request",
        "workspace/changes",
    ];

    #[test]
    fn every_known_dsh_event_type_parses_without_warnings() {
        let payload = |event_type: &str| match event_type {
            "agent-preset/selected" => r#"{"agentPreset":"standard"}"#,
            "command/run" => r#"{"commandId":"c1","name":"compact","source":{"kind":"user"}}"#,
            "command/done" => r#"{"commandId":"c1","kind":"success"}"#,
            "llm/retry" => r#"{"retry":1,"delayMs":500,"failure":{"code":"SERVER"}}"#,
            "turn/end" => r#"{"turn":1,"reason":{"kind":"completed"}}"#,
            "tool/ptc-dispatch-start" => {
                r#"{"rootCallId":"c","parentCallId":"c","subCallId":"c:ptc:1","name":"bash","arguments":{}}"#
            }
            "tool/ptc-dispatch" => {
                r#"{"rootCallId":"c","parentCallId":"c","subCallId":"c:ptc:1","name":"bash","arguments":{},"content":[]}"#
            }
            _ => "{}",
        };
        let events: Vec<String> = KNOWN_DSH_EVENT_TYPES
            .iter()
            .zip(2u64..)
            .map(|(event_type, seq)| event_line(event_type, seq, 1000, payload(event_type)))
            .collect();
        let mut lines = vec![
            versioned_header_line(4, "/tmp/p"),
            user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""hello""#),
        ];
        lines.extend(events);
        let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
        let session = parse_lines(&lines);
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn compaction_and_failed_attempt_usage_is_counted() {
        let attempt = event_line(
            "assistant/attempt",
            3,
            2000,
            r#"{"turn":1,"step":1,"stream":[{"type":"chunk","time":2000,"chunk":{"type":"usage","usage":{"inputTokens":7,"outputTokens":3}}},{"type":"chunk","time":2000,"chunk":{"type":"finish","reason":{"kind":"error"}}}]}"#,
        );
        let summary = event_line(
            "compaction/summary",
            4,
            3000,
            r#"{"compactionId":"c1","summary":[],"provider":"p","model":"m-sum","usage":{"inputTokens":100,"outputTokens":20,"cacheReadTokens":5}}"#,
        );
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &event_line(
                "request/context",
                1,
                1000,
                r#"{"provider":"p","model":"m-route","contextWindow":1000}"#,
            ),
            &user_message_line(2, 1000, r#"{"kind":"user"}"#, r#""go""#),
            &attempt,
            &summary,
        ]);
        let rows: Vec<(&str, u64, u64, u64)> = session
            .usage_events
            .iter()
            .map(|event| {
                (
                    event.model.as_str(),
                    event.input_tokens,
                    event.output_tokens,
                    event.cache_read_input_tokens,
                )
            })
            .collect();
        assert_eq!(rows, [("m-route", 7, 3, 0), ("m-sum", 100, 20, 5)]);
        assert_eq!(session.parse_warning_count, 0);

        // A fork's inherited compaction was billed to the parent.
        let fork_header = format!(
            r#"{{"type":"session","version":4,"id":"{SESSION_ID}","createdAt":1,"cwd":"/tmp/p","isSeeded":true,"parentSession":"session-parent","delegationDepth":0}}"#
        );
        let session = parse_lines(&[
            &fork_header,
            &user_message_line(2, 1000, r#"{"kind":"user"}"#, r#""go""#),
            &summary,
            &event_line("session/end-seed", 5, 3001, r#"{"inherited":true}"#),
        ]);
        assert!(session.usage_events.is_empty());
    }

    #[test]
    fn unfinished_turns_surface_as_status_rows() {
        let turn_end = |seq: u64, reason: &str| {
            event_line(
                "turn/end",
                seq,
                1000 + seq as i64,
                &format!(r#"{{"turn":1,"reason":{reason}}}"#),
            )
        };
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""go""#),
            &turn_end(2, r#"{"kind":"completed"}"#),
            &turn_end(
                3,
                r#"{"kind":"error","error":{"message":"Authentication Fails","code":"AUTH","status":401}}"#,
            ),
            &turn_end(4, r#"{"kind":"aborted","reason":{"kind":"user"}}"#),
            &turn_end(
                5,
                r#"{"kind":"aborted","reason":{"kind":"hook","reason":"stop requested"}}"#,
            ),
            &turn_end(6, r#"{"kind":"aborted","reason":{"kind":"parent"}}"#),
            &turn_end(7, r#"{"kind":"interrupted"}"#),
            &turn_end(8, r#"{"kind":"blocked"}"#),
            &turn_end(9, r#"{"kind":"max-tokens"}"#),
            &turn_end(10, r#"{"kind":"forked"}"#),
            &turn_end(11, r#"{"kind":"goal-limit"}"#),
        ]);
        let rows: Vec<&str> = session.messages[1..]
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            rows,
            [
                "[turn_failed]\nAuthentication Fails",
                "[turn_cancelled]",
                "[turn_cancelled] hook: stop requested",
                "[turn_cancelled] parent",
                "[turn_interrupted]",
                "[turn_blocked]",
                "[turn_max_tokens]",
                "[turn_ended] goal-limit",
            ]
        );
        assert!(
            session.messages[1..]
                .iter()
                .all(|m| m.role == MessageRole::System)
        );
        assert_eq!(session.parse_warning_count, 0);
    }

    fn child_links(message: &Message) -> (Vec<String>, Vec<String>) {
        let structured = message
            .tool_metadata
            .as_ref()
            .and_then(|metadata| metadata.structured.as_ref());
        let list = |key: &str| -> Vec<String> {
            structured
                .and_then(|value| value.get(key))
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default()
        };
        (list("childConversationIds"), list("childPrompts"))
    }

    #[test]
    fn delegated_children_link_to_the_tool_call_that_spawned_them() {
        let catalog = |seq: u64, child: &str, label: Option<&str>| {
            let label = label.map_or(String::new(), |label| format!(r#","label":"{label}""#));
            event_line(
                "subagent/catalog",
                seq,
                1000,
                &format!(
                    r#"{{"version":0,"childId":"{child}","childCreatedAt":1,"mode":"one-shot"{label}}}"#
                ),
            )
        };
        let agent_start = |seq: u64, child: &str, label: &str| {
            event_line(
                "tool-workflow/agent-start",
                seq,
                1000,
                &format!(r#"{{"runId":"r1","seq":{seq},"label":"{label}","childId":"{child}"}}"#),
            )
        };
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""delegate""#),
            &tool_call_line(
                2,
                1001,
                "call-sub",
                "subagent",
                r#""{\"description\":\"Count lines\"}""#,
            ),
            &catalog(3, "child-a", Some("Count lines")),
            &v4_tool_result_line(4, "call-sub", r#""12 lines""#, false, ""),
            &tool_call_line(
                5,
                1005,
                "call-wf",
                "workflow",
                r#""{\"script\":\"phase()\"}""#,
            ),
            &event_line(
                "tool-workflow/run-start",
                6,
                1006,
                r#"{"runId":"r1","name":"w"}"#,
            ),
            &catalog(7, "child-b", None),
            &agent_start(8, "child-b", "agent A"),
            &catalog(9, "child-c", None),
            &agent_start(10, "child-c", "agent B"),
            &v4_tool_result_line(11, "call-wf", r#""done""#, false, ""),
            // No delegating call is open: the child stays unlinked.
            &catalog(12, "child-d", Some("stray")),
        ]);
        let tools = tool_messages(&session);
        assert_eq!(tools.len(), 2);
        assert!(
            tools
                .iter()
                .all(|tool| tool.tool_name.as_deref() == Some("Agent"))
        );
        assert_eq!(
            child_links(tools[0]),
            (vec!["child-a".to_string()], vec!["Count lines".to_string()])
        );
        assert_eq!(
            child_links(tools[1]),
            (
                vec!["child-b".to_string(), "child-c".to_string()],
                vec!["agent A".to_string(), "agent B".to_string()]
            )
        );
        assert_eq!(tools[1].content, "done");
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn ptc_sub_calls_surface_as_their_own_tool_messages() {
        let start = |seq: u64, sub: &str, name: &str, arguments: &str| {
            event_line(
                "tool/ptc-dispatch-start",
                seq,
                1000,
                &format!(
                    r#"{{"rootCallId":"call-code","parentCallId":"call-code","subCallId":"{sub}","name":"{name}","arguments":{arguments}}}"#
                ),
            )
        };
        let settle = |seq: u64, sub: &str, name: &str, error: bool, text: &str| {
            event_line(
                "tool/ptc-dispatch",
                seq,
                1000,
                &format!(
                    r#"{{"rootCallId":"call-code","parentCallId":"call-code","subCallId":"{sub}","name":"{name}","arguments":{{"x":1}},"isError":{error},"content":[{{"type":"text","text":"{text}"}}]}}"#
                ),
            )
        };
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""run code""#),
            &tool_call_line(
                2,
                1001,
                "call-code",
                "run_code",
                r#""{\"code\":\"await tools.bash()\"}""#,
            ),
            &start(3, "call-code:ptc:1", "bash", r#"{"command":"echo ok"}"#),
            &start(
                4,
                "call-code:ptc:2",
                "read",
                r#"{"file_path":"missing.txt"}"#,
            ),
            &settle(5, "call-code:ptc:1", "bash", false, "ok"),
            &settle(6, "call-code:ptc:2", "read", true, "no such file"),
            // A settle whose start was never logged still surfaces.
            &settle(7, "call-code:ptc:3", "glob", false, "a.py"),
            &v4_tool_result_line(8, "call-code", r#""returned""#, false, ""),
        ]);
        let tools = tool_messages(&session);
        let summary: Vec<(&str, &str, Option<&str>)> = tools
            .iter()
            .map(|tool| {
                (
                    tool.tool_name.as_deref().unwrap_or(""),
                    tool.content.as_str(),
                    tool.tool_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.status.as_deref()),
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                ("run_code", "returned", Some("success")),
                ("Bash", "ok", Some("success")),
                ("Read", "no such file", Some("error")),
                ("Glob", "a.py", Some("success")),
            ]
        );
        assert_eq!(
            tools[1].tool_input.as_deref(),
            Some(r#"{"command":"echo ok"}"#)
        );
        assert_eq!(tools[3].tool_input.as_deref(), Some(r#"{"x":1}"#));
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn image_blocks_resolve_their_stored_attachment() {
        let home = TempDir::new().unwrap();
        let stored = "ab".repeat(32);
        let missing = "cd".repeat(32);
        let objects = home.path().join("attachments").join("v1").join("objects");
        std::fs::create_dir_all(objects.join("ab")).unwrap();
        std::fs::write(objects.join("ab").join(&stored), b"\x89PNG\r\n\x1a\n").unwrap();
        let session_dir = home
            .path()
            .join("sessions")
            .join("--tmp-p--")
            .join(SESSION_ID);
        std::fs::create_dir_all(&session_dir).unwrap();
        let image = |id: &str| {
            format!(
                r#"{{"type":"image","attachment":{{"attachmentId":"{id}","mediaType":"image/png","bytes":8,"width":1,"height":1}}}}"#
            )
        };
        let user = event_line(
            "user/message",
            1,
            1000,
            &format!(
                r#"{{"content":[{{"type":"text","text":"look"}},{},{},{}],"source":{{"kind":"user"}},"role":"user","id":"u1"}}"#,
                image(&format!("sha256:{stored}")),
                image(&format!("sha256:{missing}")),
                image("sha256:../../../etc/passwd"),
            ),
        );
        let log = format!("{}\n{user}\n", versioned_header_line(4, "/tmp/p"));
        let path = session_dir.join("session.v4.jsonl");
        std::fs::write(&path, log).unwrap();

        let session = parse_session_file(&path).expect("fixture must parse");
        let stored_path = objects.join("ab").join(&stored);
        assert_eq!(
            session.messages[0].content,
            format!(
                "look\n[Image: source: {}]\n[Image]\n[Image]",
                stored_path.display()
            )
        );
    }

    #[test]
    fn fallback_titles_yield_to_the_first_prompt() {
        let fallback = event_line(
            "session/title",
            3,
            1002,
            r#"{"title":"Use the bash tool","messageSeqs":[2],"source":{"kind":"fallback"}}"#,
        );
        let descriptor = event_line(
            "subagent/descriptor",
            1,
            1000,
            r#"{"version":3,"mode":"one-shot","provider":"spawn"}"#,
        );
        let prompt = user_message_line(
            2,
            1001,
            r#"{"kind":"user"}"#,
            r#""Use the bash tool exactly once to run: ls -1""#,
        );
        // An unlabeled workflow agent: the full prompt, not DSH's prefix.
        let session = parse_lines(&[
            &subagent_header_line("/tmp/p"),
            &descriptor,
            &prompt,
            &fallback,
        ]);
        assert_eq!(
            session.meta.title,
            "Use the bash tool exactly once to run: ls -1"
        );

        // An LLM title still outranks the prompt.
        let llm_title = event_line(
            "session/title",
            4,
            1003,
            r#"{"title":"List files","messageSeqs":[2],"source":{"kind":"provider","provider":"session-title-first-prompt-llm"}}"#,
        );
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &prompt,
            &fallback,
            &llm_title,
        ]);
        assert_eq!(session.meta.title, "List files");

        // Without a direct prompt, the fallback title is still a title.
        let relay = user_message_line(
            2,
            1001,
            r#"{"kind":"agent-message","form":"relay"}"#,
            r#""Agent x sent a message""#,
        );
        let session = parse_lines(&[&versioned_header_line(4, "/tmp/p"), &relay, &fallback]);
        assert_eq!(session.meta.title, "Use the bash tool");
    }

    #[test]
    fn user_attached_files_render_as_file_markers() {
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &event_line(
                "user/message",
                1,
                1000,
                r#"{"content":[{"type":"image","attachment":{"attachmentId":"sha256:00","mediaType":"image/png","bytes":1,"width":1,"height":1}},{"type":"file","attachment":{"attachmentId":"sha256:11","name":"notes.txt","mediaType":"text/plain","bytes":21}},{"type":"file","attachment":{"attachmentId":"sha256:22","bytes":3}},{"type":"text","text":"read these"}],"source":{"kind":"user"},"role":"user","id":"u1"}"#,
            ),
        ]);
        assert_eq!(
            session.messages[0].content,
            "read these\n[Image]\n[File: notes.txt]\n[File]"
        );
        assert_eq!(session.parse_warning_count, 0);
    }

    #[test]
    fn slash_commands_render_as_command_rows() {
        let run = |seq: u64, id: &str, name: &str, args: Option<&str>| {
            let args = args.map_or(String::new(), |args| format!(r#","args":"{args}""#));
            event_line(
                "command/run",
                seq,
                1000,
                &format!(
                    r#"{{"commandId":"{id}","name":"{name}"{args},"source":{{"kind":"user"}}}}"#
                ),
            )
        };
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &run(1, "c1", "compact", Some("")),
            &event_line(
                "command/done",
                2,
                1001,
                r#"{"commandId":"c1","kind":"success","text":"Compacted 38 history items (~3961 tokens)."}"#,
            ),
            &run(3, "c2", "permission", Some(" read-only")),
            &event_line(
                "command/done",
                4,
                1002,
                r#"{"commandId":"c2","kind":"success","text":"preset read-only"}"#,
            ),
            // A domain event owns this input, so the run records no args.
            &run(5, "c3", "goal", None),
            &event_line(
                "command/done",
                6,
                1003,
                r#"{"commandId":"c3","kind":"error","text":"No active goal."}"#,
            ),
            // A silent success adds no output row.
            &run(7, "c4", "plan", Some(" off")),
            &event_line(
                "command/done",
                8,
                1004,
                r#"{"commandId":"c4","kind":"success"}"#,
            ),
        ]);
        use crate::models::MessageKind::{self, CommandInput, CommandOutput};
        let rows: Vec<(Option<MessageKind>, &str)> = session
            .messages
            .iter()
            .map(|m| (m.message_kind.clone(), m.content.as_str()))
            .collect();
        assert_eq!(
            rows,
            [
                (Some(CommandInput), "/compact"),
                (
                    Some(CommandOutput),
                    "Compacted 38 history items (~3961 tokens)."
                ),
                (Some(CommandInput), "/permission read-only"),
                (Some(CommandOutput), "preset read-only"),
                (Some(CommandInput), "/goal"),
                (Some(CommandOutput), "No active goal."),
                (Some(CommandInput), "/plan off"),
            ]
        );
        assert_eq!(session.parse_warning_count, 0);

        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""hi""#),
            &event_line(
                "command/run",
                2,
                1001,
                r#"{"commandId":"c1","source":{"kind":"user"}}"#,
            ),
            &event_line("command/done", 3, 1002, r#"{"commandId":"c1","text":"?"}"#),
        ]);
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.parse_warning_count, 2);
    }

    #[test]
    fn retries_surface_as_status_rows() {
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""hi""#),
            &event_line(
                "llm/retry",
                2,
                1001,
                r#"{"retryId":"r1","turn":1,"step":1,"provider":"p","mode":"normal","policyKey":"k","retry":2,"maxRetries":5,"delayMs":947.83,"failure":{"message":"DeepSeek Messages transport failed","code":"TRANSPORT"}}"#,
            ),
            &event_line(
                "llm/retry-started",
                3,
                1002,
                r#"{"retryId":"r1","turn":1,"step":1,"retry":2}"#,
            ),
            &event_line(
                "llm/retry",
                4,
                1003,
                r#"{"retryId":"r2","turn":1,"step":1,"provider":"p","mode":"always","policyKey":"k","retry":7,"delayMs":10000,"failure":{"message":"","code":"RATE_LIMIT"}}"#,
            ),
            &event_line(
                "llm/retry",
                5,
                1004,
                r#"{"retryId":"r3","failure":{"code":"SERVER"}}"#,
            ),
        ]);
        let rows: Vec<&str> = session.messages[1..]
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            rows,
            [
                "[retry] retry 2/5 after 948 ms (TRANSPORT)\nDeepSeek Messages transport failed",
                "[retry] retry 7 after 10000 ms (RATE_LIMIT)",
            ]
        );
        // The retry without its details is a counted warning, not a row.
        assert_eq!(session.parse_warning_count, 1);
    }

    #[test]
    fn session_references_name_the_referenced_sessions() {
        let reference = |seq: u64, references: &str| {
            event_line(
                "user/message",
                seq,
                1000,
                &format!(
                    r##"{{"content":[{{"type":"text","text":"Referenced sessions\n{{\"events\":[]}}"}}],"source":{{"kind":"session-reference","form":"recall","version":1,"references":{references}}},"role":"user","id":"u{seq}"}}"##
                ),
            )
        };
        let session = parse_lines(&[
            &versioned_header_line(4, "/tmp/p"),
            &user_message_line(1, 1000, r#"{"kind":"user"}"#, r#""what did it do?""#),
            &reference(
                2,
                r#"[{"sessionId":"session-a","label":"Read image and run echo","capturedThroughSeq":30,"capturedFormatVersion":4},{"sessionId":"session-b","capturedThroughSeq":3,"capturedFormatVersion":4}]"#,
            ),
            &reference(3, "[]"),
        ]);
        let contents: Vec<&str> = session
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            contents,
            [
                "what did it do?",
                "[session_reference]\nRead image and run echo\nsession-b",
            ]
        );
        // The recalled snapshot itself never reaches the transcript.
        assert!(!session.content_text.contains("events"));
        assert_eq!(session.parse_warning_count, 1);
    }
}
