//! `MlxBackend`: agens-server's [`Backend`] on the Volundr MLX worker (`volundr-mlx/worker.py`), over a Unix socket.
//!
//! The server (the upstream `agens-server` crate, unchanged) owns the OpenAI API, the chat template, tool-call and
//! reasoning parsing, sampling defaults, stop strings and detokenization. The worker owns the model: it turns prompt
//! ids into token ids, keeps the session / prefix cache, and decides how much of each prompt it already has.
//!
//! Protocol (version 1; `volundr-mlx/NOTES_SESSION.md` has the full description). Every message is one frame: a
//! 4-byte big-endian length, then that many bytes of UTF-8 JSON (an object with a `"type"`). One connection per
//! request:
//! * client -> worker: `{"type": "generate", "prompt_ids": [...], "session_id": str | null, "sampling": {...}}`
//!   (or `{"type": "info"}`, `{"type": "stats"}`, `{"type": "clear"}`).
//! * worker -> client: zero or more `{"type": "tokens", "ids": [...]}` (one frame per decode step; several ids per
//!   step with speculative decoding), then `{"type": "done", "finish": "stop" | "length" | "cancelled", "stats":
//!   {...}}`, or `{"type": "error", "message": str}`.
//! * Closing the connection cancels the request (queued, prefilling or decoding). Requests run one at a time.

use agens_server::backend::{Backend, BackendError, SessionHint, TokenStream};
use agens_server::sampling::SamplingParams;
use serde_json::{json, Value};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::mpsc;

pub const PROTOCOL: u64 = 1;
const MAX_FRAME: usize = 256 << 20;

/// Write one frame: u32 big-endian length + JSON.
pub async fn write_frame<W: AsyncWriteExt + Unpin>(w: &mut W, v: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(v).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = u32::try_from(body.len()).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "frame too large"))?;
    w.write_all(&len.to_be_bytes()).await?;
    w.write_all(&body).await?;
    w.flush().await
}

/// Read one frame; `None` at a clean end of stream.
pub async fn read_frame<R: AsyncReadExt + Unpin>(r: &mut R) -> io::Result<Option<Value>> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len).await {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let n = u32::from_be_bytes(len) as usize;
    if n > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("frame of {n} bytes")));
    }
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).await?;
    serde_json::from_slice(&buf).map(Some).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// The `generate` frame for one request.
pub fn generate_frame(prompt_ids: &[u32], p: &SamplingParams, session: Option<&SessionHint>) -> Value {
    json!({
        "type": "generate",
        "protocol": PROTOCOL,
        "prompt_ids": prompt_ids,
        "session_id": session.map(|s| s.session_id.clone()),
        "sampling": {
            "temperature": p.temperature,
            "top_p": p.top_p,
            "top_k": p.top_k,
            "min_p": p.min_p,
            "presence_penalty": p.presence_penalty,
            "frequency_penalty": p.frequency_penalty,
            "repetition_penalty": p.repetition_penalty,
            "min_new_tokens": p.min_new_tokens,
            "max_new_tokens": p.max_new_tokens,
            "seed": p.seed,
            "logit_bias": p.logit_bias.iter().map(|(t, b)| json!([t, b])).collect::<Vec<_>>(),
            "suppress_tokens": p.suppress_tokens,
            "stop_token_ids": p.stop_token_ids,
            "ignore_eos": p.ignore_eos,
            "reasoning_budget": p.reasoning_budget,
            "reasoning_budget_soft_bias": p.reasoning_budget_soft_bias,
        },
    })
}

/// One request/response exchange with the worker (`info`, `stats`, `clear`).
pub async fn query(socket: &Path, msg: Value) -> anyhow::Result<Value> {
    let mut s = UnixStream::connect(socket).await?;
    write_frame(&mut s, &msg).await?;
    match read_frame(&mut s).await? {
        Some(v) if v["type"] == "error" => anyhow::bail!("worker: {}", v["message"]),
        Some(v) => Ok(v),
        None => anyhow::bail!("worker closed the connection"),
    }
}

pub struct MlxBackend {
    socket: PathBuf,
    context_length: usize,
}

impl MlxBackend {
    /// Connect to a worker, waiting up to `wait` for it to come up (it loads the model before it listens).
    pub async fn connect(socket: impl Into<PathBuf>, wait: Duration) -> anyhow::Result<(Self, Value)> {
        let socket = socket.into();
        let t0 = Instant::now();
        let info = loop {
            match query(&socket, json!({"type": "info"})).await {
                Ok(v) => break v,
                Err(e) if t0.elapsed() < wait => {
                    let _ = e;
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Err(e) => anyhow::bail!("no worker at {} after {:?}: {e}", socket.display(), wait),
            }
        };
        if info["protocol"].as_u64() != Some(PROTOCOL) {
            anyhow::bail!("worker speaks protocol {}, this server {PROTOCOL}", info["protocol"]);
        }
        let context_length = info["context_length"].as_u64().unwrap_or(262_144) as usize;
        Ok((MlxBackend { socket, context_length }, info))
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }
}

/// Drive one request: send it, forward each `tokens` frame, stop when the worker is done or the server drops the
/// stream (dropping the connection is the cancellation).
async fn run(
    socket: PathBuf,
    frame: Value,
    tx: &mpsc::Sender<Result<Vec<u32>, BackendError>>,
) -> Result<(), String> {
    let mut s = UnixStream::connect(&socket).await.map_err(|e| format!("connect {}: {e}", socket.display()))?;
    write_frame(&mut s, &frame).await.map_err(|e| format!("send request: {e}"))?;
    loop {
        let msg = tokio::select! {
            biased;
            _ = tx.closed() => return Ok(()),
            m = read_frame(&mut s) => m.map_err(|e| format!("worker stream: {e}"))?,
        };
        let Some(v) = msg else { return Err("worker closed the connection".into()) };
        match v["type"].as_str() {
            Some("tokens") => {
                let ids: Vec<u32> = v["ids"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_u64().map(|t| t as u32)).collect())
                    .unwrap_or_default();
                if !ids.is_empty() && tx.send(Ok(ids)).await.is_err() {
                    return Ok(());
                }
            }
            Some("done") => return Ok(()),
            Some("error") => return Err(v["message"].as_str().unwrap_or("worker error").to_string()),
            other => return Err(format!("unexpected worker frame {other:?}")),
        }
    }
}

impl Backend for MlxBackend {
    fn generate(&self, prompt_ids: Vec<u32>, sampling: SamplingParams, session: Option<SessionHint>) -> TokenStream {
        let (tx, rx) = mpsc::channel(64);
        let frame = generate_frame(&prompt_ids, &sampling, session.as_ref());
        let socket = self.socket.clone();
        tokio::spawn(async move {
            if let Err(e) = run(socket, frame, &tx).await {
                let _ = tx.send(Err(BackendError(e))).await;
            }
        });
        rx
    }

    fn context_length(&self) -> usize {
        self.context_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn frames_round_trip() {
        let (mut a, mut b) = tokio::io::duplex(1 << 16);
        let v = json!({"type": "tokens", "ids": [1, 2, 248046]});
        write_frame(&mut a, &v).await.unwrap();
        drop(a);
        assert_eq!(read_frame(&mut b).await.unwrap(), Some(v));
        assert_eq!(read_frame(&mut b).await.unwrap(), None);
    }

    #[test]
    fn generate_frame_fields() {
        let p = SamplingParams {
            temperature: 1.0,
            top_p: 0.95,
            top_k: Some(1),
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            min_new_tokens: 0,
            max_new_tokens: 16,
            seed: Some(7),
            logit_bias: vec![(5, -1.5)],
            suppress_tokens: vec![248077],
            stop_token_ids: vec![248044, 248046],
            ignore_eos: false,
            reasoning_budget: None,
            reasoning_budget_soft_bias: 0.0,
        };
        let f = generate_frame(&[1, 2, 3], &p, Some(&SessionHint { session_id: "s".into() }));
        assert_eq!(f["session_id"], "s");
        assert_eq!(f["sampling"]["top_k"], 1);
        assert_eq!(f["sampling"]["logit_bias"][0][0], 5);
        assert_eq!(f["sampling"]["stop_token_ids"][1], 248046);
        assert!(f["sampling"]["reasoning_budget"].is_null());
    }
}
