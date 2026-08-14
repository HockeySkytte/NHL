//! Signed-session cookie helpers.
//!
//! Format: `base64url(json).base64url(hmac-sha256(secret, base64url))`.
//! The payload is an object:
//! ```json
//! { "auth_user": { ... } | null, "csrf_token": "...", "flashes": [...] }
//! ```
//! This is the Rust app's own session cookie (not Flask-compatible) — it is
//! only read/written by the Rust service.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Decoded session payload carried in the `session` cookie.
#[derive(Clone, Debug, Default)]
pub struct SessionData {
    pub auth_user: Option<Value>,
    pub csrf_token: String,
    pub flashes: Vec<Value>,
}

impl SessionData {
    pub fn encode(&self, secret: &str) -> String {
        let payload = json!({
            "auth_user": self.auth_user.clone().unwrap_or(Value::Null),
            "csrf_token": self.csrf_token,
            "flashes": self.flashes,
        });
        encode_session(secret, &payload)
    }

    pub fn from_cookie(secret: &str, cookie_value: &str) -> Option<Self> {
        let payload = decode_session(secret, cookie_value)?;
        let auth_user = match payload.get("auth_user") {
            Some(v) if !v.is_null() => Some(v.clone()),
            _ => None,
        };
        let csrf_token = payload
            .get("csrf_token")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let flashes = payload
            .get("flashes")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Some(Self {
            auth_user,
            csrf_token,
            flashes,
        })
    }
}

fn sign(secret: &str, payload_b64: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("hmac accepts any key length");
    mac.update(payload_b64.as_bytes());
    URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes())
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Encode a session payload into a cookie value.
pub fn encode_session(secret: &str, payload: &Value) -> String {
    let b64 = URL_SAFE_NO_PAD.encode(payload.to_string().as_bytes());
    let sig = sign(secret, &b64);
    format!("{b64}.{sig}")
}

/// Decode + verify a session cookie value.
pub fn decode_session(secret: &str, cookie: &str) -> Option<Value> {
    let (b64, sig) = cookie.rsplit_once('.')?;
    let expected = sign(secret, b64);
    if !ct_eq(expected.as_bytes(), sig.as_bytes()) {
        return None;
    }
    let bytes = URL_SAFE_NO_PAD.decode(b64).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Extract the `session` cookie value from a Cookie header.
pub fn session_cookie_value(cookie_header: &str) -> Option<String> {
    cookie_header
        .split(';')
        .find_map(|c| {
            let mut it = c.trim().splitn(2, '=');
            if it.next()? == "session" {
                let v = it.next()?;
                if !v.is_empty() {
                    Some(v.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_and_tamper() {
        let secret = "test-secret";
        let payload = json!({"email": "a@b.c", "is_admin": true});
        let cookie = encode_session(secret, &payload);
        let decoded = decode_session(secret, &cookie).unwrap();
        assert_eq!(decoded["email"], "a@b.c");
        assert_eq!(decoded["is_admin"], true);

        // Tampered payload must fail verification.
        let mut tampered = cookie.clone();
        let idx = tampered.find('.').unwrap();
        let b64 = &tampered[..idx];
        let mut bytes = URL_SAFE_NO_PAD.decode(b64).unwrap();
        bytes[0] ^= 1;
        let new_b64 = URL_SAFE_NO_PAD.encode(&bytes);
        tampered = format!("{new_b64}.{}", &tampered[idx + 1..]);
        assert!(decode_session(secret, &tampered).is_none());

        // Wrong secret must fail.
        assert!(decode_session("other-secret", &cookie).is_none());
    }

    #[test]
    fn session_data_roundtrip() {
        let secret = "test-secret";
        let data = SessionData {
            auth_user: Some(json!({"user_id": "u1", "email": "a@b.c"})),
            csrf_token: "abc123".to_string(),
            flashes: vec![json!(["success", "Hello"])],
        };
        let cookie = data.encode(secret);
        let decoded = SessionData::from_cookie(secret, &cookie).unwrap();
        assert_eq!(decoded.auth_user.unwrap()["user_id"], "u1");
        assert_eq!(decoded.csrf_token, "abc123");
        assert_eq!(decoded.flashes.len(), 1);

        // No auth_user -> None
        let anon = SessionData::default().encode(secret);
        let decoded_anon = SessionData::from_cookie(secret, &anon).unwrap();
        assert!(decoded_anon.auth_user.is_none());
    }
}

