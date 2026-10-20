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
//! * Closing the connection cancels the request (queued, prefilling or decoding). Every request has its own
//!   connection, so concurrent requests reach the worker together; the worker batches them (continuous batching,
//!   `volundr-mlx/scheduler.py`, NOTES_BATCHED_SERVE.md).

use agens_server::backend::{Backend, BackendError, SessionHint, TokenStream};
use agens_server::sampling::SamplingParams;
use serde_json::{json, Value};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, watch};

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

tokio::task_local! {
    /// The HTTP request's disconnect signal, set by [`cancel_on_disconnect`] around each request: the sender lives
    /// in the request future and, for a streamed answer, in its response body, so it is dropped when hyper drops
    /// either (the client went away). `generate` runs inside that future and picks the receiver up.
    static REQUEST_GONE: watch::Receiver<()>;
}

/// SSE comment sent when a streamed answer has had no data for this long (queued, or prefilling: about a minute at
/// 20K new tokens). The write is how hyper finds out that a client disconnected; without it agens-server would only
/// notice at the first token, after the whole prefill. Clients ignore SSE comments.
pub const KEEPALIVE: Duration = Duration::from_secs(2);

async fn gone(rx: &mut Option<watch::Receiver<()>>) {
    match rx {
        // nothing is ever sent: changed() returns an error only when the sender (the request) is dropped
        Some(r) => while r.changed().await.is_ok() {},
        None => std::future::pending::<()>().await,
    }
}

/// axum middleware: run the request with a disconnect signal ([`REQUEST_GONE`]) and keep streamed (SSE) answers
/// alive with comment lines while they have nothing to send. Dropping the request future or the response body (what
/// hyper does when the client disconnects) cancels the backend request.
pub async fn cancel_on_disconnect(req: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    use futures_util::StreamExt;
    let (tx, rx) = watch::channel(());
    let resp = REQUEST_GONE.scope(rx, next.run(req)).await;
    let sse = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    if !sse {
        return resp; // (the request is over: dropping tx is harmless)
    }
    let (parts, body) = resp.into_parts();
    let stream = futures_util::stream::unfold((body.into_data_stream(), tx), |(mut s, tx)| async move {
        tokio::select! {
            item = s.next() => item.map(|b| (b, (s, tx))),
            _ = tokio::time::sleep(KEEPALIVE) => {
                Some((Ok(axum::body::Bytes::from_static(b": keep-alive\n\n")), (s, tx)))
            }
        }
    });
    axum::response::Response::from_parts(parts, axum::body::Body::from_stream(stream))
}

/// Drive one request: send it, forward each `tokens` frame, stop when the worker is done, the server drops the
/// stream or the HTTP client disconnects (closing the worker connection is the cancellation).
async fn run(
    socket: PathBuf,
    frame: Value,
    tx: &mpsc::Sender<Result<Vec<u32>, BackendError>>,
    mut client_gone: Option<watch::Receiver<()>>,
) -> Result<(), String> {
    let mut s = UnixStream::connect(&socket).await.map_err(|e| format!("connect {}: {e}", socket.display()))?;
    write_frame(&mut s, &frame).await.map_err(|e| format!("send request: {e}"))?;
    loop {
        let msg = tokio::select! {
            biased;
            _ = tx.closed() => return Ok(()),
            _ = gone(&mut client_gone) => return Ok(()),
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
        let client_gone = REQUEST_GONE.try_with(|r| r.clone()).ok();
        tokio::spawn(async move {
            if let Err(e) = run(socket, frame, &tx, client_gone).await {
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

    fn params(max_new_tokens: usize) -> SamplingParams {
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: Some(1),
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            min_new_tokens: 0,
            max_new_tokens,
            seed: None,
            logit_bias: vec![],
            suppress_tokens: vec![],
            stop_token_ids: vec![9],
            ignore_eos: false,
            reasoning_budget: None,
            reasoning_budget_soft_bias: 0.0,
        }
    }

    /// A fake worker: answers `info`, then streams `steps` for one generate and reports whether the client closed
    /// the connection before it was done.
    async fn fake_worker(path: PathBuf, steps: Vec<Vec<u32>>, delay: Duration) -> tokio::task::JoinHandle<bool> {
        let l = tokio::net::UnixListener::bind(&path).unwrap();
        tokio::spawn(async move {
            loop {
                let (mut s, _) = l.accept().await.unwrap();
                let m = read_frame(&mut s).await.unwrap().unwrap();
                if m["type"] == "info" {
                    write_frame(&mut s, &json!({"type": "info", "protocol": PROTOCOL, "context_length": 4096}))
                        .await
                        .unwrap();
                    continue;
                }
                assert_eq!(m["prompt_ids"], json!([1, 2, 3]));
                tokio::time::sleep(delay).await; // "prefill"
                for st in &steps {
                    if write_frame(&mut s, &json!({"type": "tokens", "ids": st})).await.is_err() {
                        return true;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                let _ = write_frame(&mut s, &json!({"type": "done", "finish": "stop", "stats": {}})).await;
                // closed by the client? (a read sees EOF)
                return matches!(read_frame(&mut s).await, Ok(None) | Err(_));
            }
        })
    }

    #[tokio::test]
    async fn backend_streams_steps_and_cancels_on_drop() {
        let dir = std::env::temp_dir().join(format!("vs-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.sock");
        let _ = std::fs::remove_file(&path);
        let worker = fake_worker(path.clone(), vec![vec![5], vec![6, 7], vec![9]], Duration::from_millis(1500)).await;
        let (b, info) = MlxBackend::connect(&path, Duration::from_secs(5)).await.unwrap();
        assert_eq!(info["context_length"], 4096);
        assert_eq!(b.context_length(), 4096);
        let mut rx = b.generate(vec![1, 2, 3], params(16), None);
        let mut got = Vec::new();
        while let Some(step) = rx.recv().await {
            got.extend(step.unwrap());
        }
        assert_eq!(got, vec![5, 6, 7, 9]);
        assert!(worker.await.unwrap());

        // dropping the stream closes the connection: the worker's next write fails
        let _ = std::fs::remove_file(&path);
        let worker = fake_worker(path.clone(), (0..200).map(|i| vec![100 + i]).collect(), Duration::ZERO).await;
        let mut rx = b.generate(vec![1, 2, 3], params(1000), None);
        let first = rx.recv().await.unwrap().unwrap();
        assert_eq!(first, vec![100]);
        drop(rx);
        let t = Instant::now();
        assert!(tokio::time::timeout(Duration::from_secs(5), worker).await.unwrap().unwrap());
        assert!(t.elapsed() < Duration::from_secs(2));

        // the HTTP request going away (its disconnect signal dropped) cancels a request that is still "prefilling"
        let _ = std::fs::remove_file(&path);
        let worker = fake_worker(path.clone(), vec![vec![1]], Duration::from_secs(30)).await;
        let (gone_tx, gone_rx) = watch::channel(());
        let mut rx = REQUEST_GONE.scope(gone_rx, async { b.generate(vec![1, 2, 3], params(16), None) }).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        drop(gone_tx);
        let t = Instant::now();
        assert!(tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap().is_none());
        assert!(t.elapsed() < Duration::from_secs(1));
        worker.abort();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Concurrent requests are not serialised here: each `generate` has its own worker connection, so two streams
    /// are in flight at once (the worker batches them). A fake worker that serves every connection on its own task
    /// streams 20 steps per request; the second request's first step must arrive while the first is still streaming.
    #[tokio::test]
    async fn concurrent_requests_stream_together() {
        let dir = std::env::temp_dir().join(format!("vs-conc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.sock");
        let _ = std::fs::remove_file(&path);
        let l = tokio::net::UnixListener::bind(&path).unwrap();
        let open = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (o2, p2) = (open.clone(), peak.clone());
        let srv = tokio::spawn(async move {
            loop {
                let (mut s, _) = l.accept().await.unwrap();
                let (o, p) = (o2.clone(), p2.clone());
                tokio::spawn(async move {
                    let m = read_frame(&mut s).await.unwrap().unwrap();
                    if m["type"] == "info" {
                        let _ = write_frame(&mut s, &json!({"type": "info", "protocol": PROTOCOL, "context_length": 4096})).await;
                        return;
                    }
                    let now = o.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    p.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                    for i in 0..20u32 {
                        let _ = write_frame(&mut s, &json!({"type": "tokens", "ids": [i]})).await;
                        tokio::time::sleep(Duration::from_millis(25)).await;
                    }
                    let _ = write_frame(&mut s, &json!({"type": "done", "finish": "length", "stats": {}})).await;
                    o.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                });
            }
        });
        let (b, _) = MlxBackend::connect(&path, Duration::from_secs(5)).await.unwrap();
        let t0 = Instant::now();
        let mut a = b.generate(vec![1, 2, 3], params(20), None);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut c = b.generate(vec![1, 2, 3], params(20), None);
        let first_c = { c.recv().await.unwrap().unwrap(); t0.elapsed() };
        let (mut na, mut nc) = (0, 1);
        let mut last_a = Duration::ZERO;
        loop {
            tokio::select! {
                x = a.recv() => match x { Some(s) => { s.unwrap(); na += 1; last_a = t0.elapsed(); } None => break },
                x = c.recv() => if x.is_some() { nc += 1 },
            }
        }
        while c.recv().await.is_some() {
            nc += 1;
        }
        assert_eq!((na, nc), (20, 20));
        assert!(first_c < last_a, "second stream started at {first_c:?}, after the first ended at {last_a:?}");
        assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 2, "both requests at the worker at once");
        srv.abort();
        let _ = std::fs::remove_dir_all(&dir);
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
