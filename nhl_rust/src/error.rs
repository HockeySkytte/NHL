//! Unified API error type.
//!
//! The Flask app has no custom error handlers, so most failure modes map to
//! plain 500s there. We keep JSON bodies (`{"error": ...}`) like the Flask API
//! routes use for their explicit error responses.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    BadRequest(serde_json::Value),
    Internal(String),
    Template(String),
    Io(std::io::Error),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::NotFound(m) => write!(f, "not found: {m}"),
            ApiError::BadRequest(v) => write!(f, "bad request: {v}"),
            ApiError::Internal(m) => write!(f, "internal error: {m}"),
            ApiError::Template(m) => write!(f, "template error: {m}"),
            ApiError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for ApiError {}

impl From<std::io::Error> for ApiError {
    fn from(e: std::io::Error) -> Self {
        ApiError::Io(e)
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        ApiError::Internal(format!("json error: {e}"))
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::NotFound(msg) => (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": msg })),
            )
                .into_response(),
            ApiError::BadRequest(payload) => (StatusCode::BAD_REQUEST, Json(payload)).into_response(),
            ApiError::Internal(_) | ApiError::Template(_) | ApiError::Io(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "internal_error" })),
            )
                .into_response(),
        }
    }
}
