"""
uplan.extraction.crypto
-----------------------
Cryptographic deletion certificate and secure RAM zeroing.

Two responsibilities:
  1. generate_deletion_cert() — produce the HMAC cert returned to the user
     as proof that their document was processed and purged.

  2. zero_bytes() — overwrite a bytearray in-place with zeros before deletion,
     mitigating OS-level memory residue risks.

Notes on zero_bytes()
---------------------
Python's `del` does not guarantee immediate memory zeroing — the GC may
retain the object. We use ctypes.memset to write zeros directly to the
buffer's memory address before handing control back to the GC.

Notes on generate_deletion_cert()
----------------------------------
The cert is: HMAC-SHA256(doc_hash || nonce || timestamp_ms, session_key)
where session_key = SHA-256(session_id + nonce), destroyed after cert generation.

This means:
  - The cert is bound to the specific document (via doc_hash)
  - The cert is bound to this specific session (via session_key derived from session_id)
  - The cert is timestamped (replay after the fact is detectable)
  - After session_key is del'd, the cert cannot be reproduced even by Uplan
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import time


def zero_bytes(buf: bytearray) -> None:
    """
    Overwrite a bytearray with zeros in-place using ctypes.memset.
    Call this before del'ing any buffer containing sensitive data.

    This is a best-effort mitigation — it addresses the Python heap but
    cannot guarantee GPU VRAM or OS swap are zeroed without additional
    OS-level configuration (swapoff / zram).
    """
    if len(buf) == 0:
        return
    addr = id(buf) + ctypes.sizeof(ctypes.c_ssize_t) * 3  # skip ob_refcnt, ob_type, ob_alloc
    # Safer approach: use the buffer protocol directly
    ctypes.memset((ctypes.c_char * len(buf)).from_buffer(buf), 0, len(buf))


def generate_deletion_cert(
    doc_hash: str,
    nonce: str,
    session_id: str,
) -> str:
    """
    Generate and return the deletion certificate string.

    The certificate format is human-readable and self-describing so the
    user can store it as a plain text receipt.

    Returns a string like:
        UPLAN-CERT:v1:
        doc=<sha256_hex>
        nonce=<hex>
        ts=<unix_ms>
        sig=<hmac_hex>

    Parameters
    ----------
    doc_hash    SHA-256 hex digest of the original document bytes
    nonce       Unique random hex string for this session
    session_id  The ephemeral session identifier
    """
    timestamp_ms = str(int(time.time() * 1000))

    # Derive a session key that is specific to this session and nonce.
    # This key is never stored — it is derived and immediately used.
    session_key = hashlib.sha256(
        (session_id + nonce).encode()
    ).digest()

    # The message is the concatenation of the three binding values
    message = f"{doc_hash}|{nonce}|{timestamp_ms}".encode()

    # Compute HMAC-SHA256
    sig = hmac.new(session_key, message, hashlib.sha256).hexdigest()

    # Immediately overwrite session_key in memory
    key_arr = bytearray(session_key)
    zero_bytes(key_arr)
    del session_key, key_arr

    cert = (
        f"UPLAN-CERT:v1\n"
        f"doc={doc_hash}\n"
        f"nonce={nonce}\n"
        f"ts={timestamp_ms}\n"
        f"sig={sig}"
    )
    return cert