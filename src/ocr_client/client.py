"""
Halotec OCR Service HTTP Client.
Complies with OCR_SERVICE_DOCUMENTATION.md & TDD v1.9:
- Strictly numeric gate string: gate="1" (non-numeric triggers 500 error on backend).
- Base64-encoded image payloads.
- Sends required fields: id, gate, image, camera_position, camera_type.
- Parses nested DataDetection response: results.container_number, results.iso_code.
- Automatic JWT refresh via GET /refresh_token.
- Thread-safe and resilient with timeout and retry logic.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from src.utils.security import is_jwt_format, retrieve_auth_token, store_auth_token


class OcrClientError(Exception):
    """Base exception for OCR Client failures."""
    pass


class HalotecOcrClient:
    VALID_CAMERA_POSITIONS = {"left", "right", "rear", "top"}

    def __init__(
        self,
        base_url: str = "https://your-ocr-service.example.com",
        gate: str = "1",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        # STRICT RULE: Gate MUST be a numeric string
        self.gate = str(int(gate)) if str(gate).isdigit() else "1"
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    def get_token(self) -> Optional[str]:
        return retrieve_auth_token()

    def set_token(self, token: str) -> None:
        store_auth_token(token)

    def _get_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        current_token = token or self.get_token()
        headers = {"Content-Type": "application/json"}
        if current_token:
            headers["Authorization"] = f"Bearer {current_token}"
        return headers

    def login(self, username: str, password: str) -> str:
        """
        Authenticates via POST /login (application/x-www-form-urlencoded).
        Stores token securely in Windows DPAPI upon success.
        """
        url = f"{self.base_url}/login"
        payload = {
            "username": username,
            "password": password,
            "grant_type": "password",
        }
        try:
            resp = self._session.post(
                url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("access_token")
                if token:
                    self.set_token(token)
                    return token
                raise OcrClientError("Login succeeded but no access_token in response.")
            elif resp.status_code in (400, 401):
                raise OcrClientError("Username atau password salah.")
            else:
                raise OcrClientError(f"Login failed (HTTP {resp.status_code}): {resp.text}")
        except requests.RequestException as e:
            raise OcrClientError(f"Gagal menghubungi server OCR: {e}")

    def refresh_token(self, token_override: Optional[str] = None) -> Optional[str]:
        """Calls GET /refresh_token using the provided or current bearer token."""
        token = token_override or self.get_token()
        if not token:
            return None

        clean_token = token.strip()
        if clean_token.lower().startswith("bearer "):
            clean_token = clean_token[7:].strip()
        clean_token = clean_token.strip("'\"")

        if not is_jwt_format(clean_token):
            return None

        url = f"{self.base_url}/refresh_token"
        headers = {"Authorization": f"Bearer {clean_token}"}

        try:
            resp = self._session.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                new_token = data.get("access_token")
                if new_token:
                    self.set_token(new_token)
                    return new_token
        except Exception:
            pass
        return None

    def validate_token(self, candidate_token: str) -> Tuple[bool, str, Optional[str]]:
        """
        Strict validation of whether candidate_token is genuine and accepted by Halotec OCR API.
        Returns: (is_valid, message, refreshed_token)
        """
        clean_token = candidate_token.strip()
        if clean_token.lower().startswith("bearer "):
            clean_token = clean_token[7:].strip()
        clean_token = clean_token.strip("'\"")

        if not clean_token:
            return False, "Token tidak boleh kosong.", None

        if not is_jwt_format(clean_token):
            return False, "Format token tidak valid! Token harus berupa JWT Bearer Token standar (3 segmen dipisahkan titik).", None

        url = f"{self.base_url}/refresh_token"
        headers = {"Authorization": f"Bearer {clean_token}"}

        try:
            resp = self._session.get(url, headers=headers, timeout=min(15, self.timeout))
            if resp.status_code == 200:
                data = resp.json()
                new_token = data.get("access_token") or clean_token
                return True, "Token valid dan aktif.", new_token
            elif resp.status_code in (401, 403):
                return False, "Token ditolak: Sesi kedaluwarsa atau token tidak terdaftar.", None
            elif resp.status_code == 500:
                return False, "Token ditolak oleh server Halotec OCR (Signature / Payload JWT tidak dikenali).", None
            else:
                return False, f"Server menolak token dengan status HTTP {resp.status_code}.", None
        except requests.Timeout:
            return False, "Koneksi ke server Halotec OCR timeout saat memverifikasi token.", None
        except requests.RequestException as e:
            return False, f"Gagal menghubungi server Halotec OCR: {e}", None

    def recognize_image_file(
        self,
        image_path: Path | str,
        detection_type: str = "container_number",
        camera_position: str = "rear",
        camera_type: str = "normal",
        req_id: int = 1,
        log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        context_info: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Sends an image crop to POST /api/ocr.
        Returns: (detected_text, confidence)
        """
        path = Path(image_path)
        if not path.is_file():
            raise OcrClientError(f"Image crop file not found: {path}")

        with open(path, "rb") as f:
            encoded_bytes = base64.b64encode(f.read())
            base64_str = encoded_bytes.decode("utf-8")

        return self.recognize_base64(
            base64_str=base64_str,
            detection_type=detection_type,
            camera_position=camera_position,
            camera_type=camera_type,
            req_id=req_id,
            log_callback=log_callback,
            context_info=context_info,
        )

    def recognize_base64(
        self,
        base64_str: str,
        detection_type: str = "container_number",
        camera_position: str = "rear",
        camera_type: str = "normal",
        req_id: int = 1,
        log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        context_info: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Sends Base64 image payload to Halotec OCR API (POST /api/ocr).
        Adheres to OpenAPI schema: id, gate, image, camera_position, camera_type.
        Includes live logging callbacks and rate-limit backoff.
        """
        cam_pos = camera_position.lower().strip()
        if cam_pos not in self.VALID_CAMERA_POSITIONS:
            cam_pos = "rear"

        url = f"{self.base_url}/api/ocr"
        payload = {
            "id": req_id,
            "gate": self.gate,  # Numeric string "1"
            "image": base64_str,
            "camera_position": cam_pos,
            "camera_type": camera_type,
            # IMPORTANT: send the requested OCR target to the API.
            # Previously detection_type was only used locally to parse the response,
            # so the server could default to container_number even for ISO crops.
            "detection_type": detection_type,
        }

        ctx = context_info or f"Req-{req_id}"
        payload_kb = round(len(base64_str) * 3 / 4 / 1024, 1)
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            headers = self._get_headers()
            req_start = time.time()

            if log_callback:
                try:
                    log_callback({
                        "event": "REQUEST_SENDING",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "req_id": req_id,
                        "context_info": ctx,
                        "url": url,
                        "gate": self.gate,
                        "detection_type": detection_type,
                        "camera_position": cam_pos,
                        "payload_kb": payload_kb,
                    })
                except Exception:
                    pass

            try:
                resp = self._session.post(url, json=payload, headers=headers, timeout=self.timeout)
                duration_ms = round((time.time() - req_start) * 1000, 1)

                if resp.status_code == 401 and attempt == 1:
                    # Token expired, try refreshing once
                    if log_callback:
                        try:
                            log_callback({
                                "event": "RESPONSE_RECEIVED",
                                "status": "UNAUTHORIZED_REFRESHING",
                                "status_code": 401,
                                "duration_ms": duration_ms,
                                "attempt": attempt,
                                "server_message": "Token expired, refreshing token...",
                                "context_info": ctx,
                                "detection_type": detection_type,
                            })
                        except Exception:
                            pass

                    new_token = self.refresh_token()
                    if new_token:
                        headers = self._get_headers(new_token)
                        req_start = time.time()
                        resp = self._session.post(url, json=payload, headers=headers, timeout=self.timeout)
                        duration_ms = round((time.time() - req_start) * 1000, 1)

                if resp.status_code == 200:
                    data = resp.json()
                    text = ""
                    confidence = 0.0

                    if isinstance(data, dict):
                        # API versions may return ISO text under results.iso_code,
                        # or under the generic container_number/container_detection
                        # key even when the requested crop is an ISO-size code.
                        results = data.get("results") or data.get("data") or {}
                        if not isinstance(results, dict):
                            results = {}

                        def _as_items(value):
                            if value is None:
                                return []
                            if isinstance(value, list):
                                return value
                            if isinstance(value, dict):
                                return [value]
                            if isinstance(value, str):
                                return [{"text": value}]
                            return []

                        def _best_text(items):
                            valid_items = [x for x in items if isinstance(x, dict) and str(x.get("text", "")).strip()]
                            if not valid_items:
                                return "", 0.0
                            def _score(x):
                                try:
                                    return float(x.get("score_text", x.get("score_bbox", x.get("confidence", 0.0))) or 0.0)
                                except (TypeError, ValueError):
                                    return 0.0
                            best_item = max(valid_items, key=_score)
                            try:
                                score = _score(best_item)
                            except Exception:
                                score = 0.0
                            return str(best_item.get("text", "")).strip(), score

                        # Preferred key for the requested target.
                        target_keys = [detection_type]
                        if detection_type == "iso_code":
                            # Compatibility fallback for OCR services that ignore
                            # detection_type and return their generic text bucket.
                            target_keys += ["container_number", "container_detection", "text_detection"]
                        else:
                            target_keys += ["container_detection"]

                        for key in target_keys:
                            candidate_text, candidate_conf = _best_text(_as_items(results.get(key)))
                            if candidate_text:
                                text, confidence = candidate_text, candidate_conf
                                break

                        # Flat and nested fallback fields used by different API versions.
                        if not text:
                            flat_keys = (["iso_code", "text", "container_number"] if detection_type == "iso_code"
                                         else ["text", "container_number"])
                            for key in flat_keys:
                                value = data.get(key)
                                if isinstance(value, str) and value.strip():
                                    text = value.strip()
                                    break
                                if isinstance(value, (list, dict)):
                                    candidate_text, candidate_conf = _best_text(_as_items(value))
                                    if candidate_text:
                                        text, confidence = candidate_text, candidate_conf
                                        break
                            try:
                                confidence = float(data.get("confidence", confidence) or confidence or 0.0)
                            except (TypeError, ValueError):
                                pass

                        # Normalize ISO size/type code (e.g. 45G1).
                        if text and detection_type == "iso_code":
                            import re
                            upper_text = text.upper().replace(" ", "")
                            m = re.findall(r"[0-9A-Z]{2}[GPRUTV][0-9A-Z]", upper_text)
                            if m:
                                text = m[0]
                            else:
                                m4 = re.findall(r"[0-9A-Z]{4}", upper_text)
                                if m4:
                                    text = m4[0]

                    server_msg = "OK"
                    if isinstance(data, dict) and data.get("message"):
                        server_msg = str(data["message"])

                    if log_callback:
                        try:
                            log_callback({
                                "event": "RESPONSE_RECEIVED",
                                "status": "SUCCESS",
                                "status_code": 200,
                                "duration_ms": duration_ms,
                                "attempt": attempt,
                                "server_message": server_msg,
                                "text": text,
                                "confidence": confidence,
                                "context_info": ctx,
                                "detection_type": detection_type,
                            })
                        except Exception:
                            pass

                    return text, confidence

                elif resp.status_code == 429:
                    server_msg = resp.text.strip()
                    try:
                        rj = resp.json()
                        if isinstance(rj, dict):
                            server_msg = rj.get("message") or rj.get("detail") or str(rj)
                    except Exception:
                        pass

                    if log_callback:
                        try:
                            log_callback({
                                "event": "RESPONSE_RECEIVED",
                                "status": "RATE_LIMIT",
                                "status_code": 429,
                                "duration_ms": duration_ms,
                                "attempt": attempt,
                                "server_message": server_msg,
                                "context_info": ctx,
                                "detection_type": detection_type,
                            })
                        except Exception:
                            pass

                    if attempt < self.max_retries:
                        time.sleep(2.0 * attempt)
                        continue
                    raise OcrClientError(f"OCR API Rate Limit (429): {server_msg}")

                elif resp.status_code == 401:
                    if log_callback:
                        try:
                            log_callback({
                                "event": "RESPONSE_RECEIVED",
                                "status": "UNAUTHORIZED",
                                "status_code": 401,
                                "duration_ms": duration_ms,
                                "attempt": attempt,
                                "server_message": resp.text.strip(),
                                "context_info": ctx,
                                "detection_type": detection_type,
                            })
                        except Exception:
                            pass
                    raise OcrClientError("Authentication failed (401 Unauthorized). Please check API token.")

                elif resp.status_code == 500:
                    if log_callback:
                        try:
                            log_callback({
                                "event": "RESPONSE_RECEIVED",
                                "status": "SERVER_ERROR",
                                "status_code": 500,
                                "duration_ms": duration_ms,
                                "attempt": attempt,
                                "server_message": resp.text.strip(),
                                "context_info": ctx,
                                "detection_type": detection_type,
                            })
                        except Exception:
                            pass
                    raise OcrClientError(f"OCR Server Internal Error (500): {resp.text}")

                else:
                    if log_callback:
                        try:
                            log_callback({
                                "event": "RESPONSE_RECEIVED",
                                "status": "CLIENT_ERROR",
                                "status_code": resp.status_code,
                                "duration_ms": duration_ms,
                                "attempt": attempt,
                                "server_message": resp.text.strip(),
                                "context_info": ctx,
                                "detection_type": detection_type,
                            })
                        except Exception:
                            pass
                    raise OcrClientError(f"OCR API returned HTTP {resp.status_code}: {resp.text}")

            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                duration_ms = round((time.time() - req_start) * 1000, 1)
                if log_callback:
                    try:
                        log_callback({
                            "event": "RESPONSE_RECEIVED",
                            "status": "NETWORK_ERROR",
                            "status_code": 0,
                            "duration_ms": duration_ms,
                            "attempt": attempt,
                            "server_message": str(e),
                            "context_info": ctx,
                            "detection_type": detection_type,
                        })
                    except Exception:
                        pass
                time.sleep(1.0 * attempt)
            except OcrClientError:
                raise
            except Exception as e:
                last_error = e
                duration_ms = round((time.time() - req_start) * 1000, 1)
                if log_callback:
                    try:
                        log_callback({
                            "event": "RESPONSE_RECEIVED",
                            "status": "CLIENT_ERROR",
                            "status_code": 0,
                            "duration_ms": duration_ms,
                            "attempt": attempt,
                            "server_message": str(e),
                            "context_info": ctx,
                            "detection_type": detection_type,
                        })
                    except Exception:
                        pass
                time.sleep(1.0 * attempt)

        raise OcrClientError(f"OCR API request failed after {self.max_retries} attempts: {last_error}")
