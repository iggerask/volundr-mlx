//! `volundr-server`: the upstream agens-server HTTP front end (OpenAI API, chat template, parsers) serving the
//! Volundr MLX worker. Start the worker first (`volundr-mlx/worker.py`, or `./serve.sh` for both).

use agens_server::server::{router, AppState};
use anyhow::{bail, Context, Result};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use volundr_server::{cancel_on_disconnect, query, MlxBackend};

const USAGE: &str = "usage:
  volundr-server --model-dir <dir> [--socket PATH] [--host 127.0.0.1] [--port 30000]
                 [--served-model-name NAME] [--context-length N] [--wait-secs 900]

  <dir>: tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json, config.json
         (volundr-mlx/models/volundr-int4-mlx has them)
  --socket: the worker's Unix socket (default $VOLUNDR_WORKER_SOCKET, else $TMPDIR/volundr-mlx-worker.sock)
  GET /session_cache returns the worker's cache statistics.";

pub fn default_socket() -> PathBuf {
    if let Ok(s) = std::env::var("VOLUNDR_WORKER_SOCKET") {
        return PathBuf::from(s);
    }
    std::env::temp_dir().join("volundr-mlx-worker.sock")
}

async fn session_cache(State(socket): State<Arc<PathBuf>>) -> impl IntoResponse {
    match query(&socket, serde_json::json!({"type": "stats"})).await {
        Ok(v) => (StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], v.to_string()),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::json!({"error": e.to_string()}).to_string(),
        ),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_dir = None;
    let mut socket = default_socket();
    let mut host = "127.0.0.1".to_string();
    let mut port = 30000u16;
    let mut name = None;
    let mut context_length = None;
    let mut wait = 900u64;
    let mut i = 0;
    while i < args.len() {
        let val = || args.get(i + 1).with_context(|| format!("{} needs a value\n{USAGE}", args[i]));
        match args[i].as_str() {
            "--model-dir" | "--model-path" => model_dir = Some(val()?.clone()),
            "--socket" => socket = PathBuf::from(val()?),
            "--host" => host = val()?.clone(),
            "--port" => port = val()?.parse()?,
            "--served-model-name" => name = Some(val()?.clone()),
            "--context-length" => context_length = Some(val()?.parse::<usize>()?),
            "--wait-secs" => wait = val()?.parse()?,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            o => bail!("unknown flag {o}\n{USAGE}"),
        }
        i += 2;
    }
    let model_dir = model_dir.context(USAGE)?;
    eprintln!("volundr-server: waiting for the MLX worker at {}", socket.display());
    let (backend, info) = MlxBackend::connect(&socket, Duration::from_secs(wait)).await?;
    eprintln!("volundr-server: worker up: {info}");
    let mut state = AppState::from_model_dir(&model_dir, Arc::new(backend), name)?;
    if let Some(n) = context_length {
        state.context_length = n;
    }
    let model_name = state.model_name.clone();
    let app = router(Arc::new(state))
        .merge(Router::new().route("/session_cache", get(session_cache)).with_state(Arc::new(socket)))
        .layer(axum::middleware::from_fn(cancel_on_disconnect));
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
    eprintln!("volundr-server: serving {model_name} on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
