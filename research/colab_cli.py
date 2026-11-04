"""Run the official Colab CLI with the owner's approved scope subset (Issue #7).

Use the WSL google-colab-cli virtual-environment Python.  ``--refresh-auth``
before a normal command forces a refresh with this same restricted subset.
No credentials, OAuth client configuration, or runtime endpoints belong here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

# This file's name would otherwise shadow the installed namespace package.
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path[:] = [
    value for value in sys.path
    if Path(value or os.curdir).resolve() != _SCRIPT_DIRECTORY
]

APPROVED_SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/colaboratory",
)
# tokeninfo expands the basic identity scopes to these equivalent OIDC aliases.
APPROVED_TOKEN_SCOPES = frozenset(APPROVED_SCOPES) | {"email", "profile"}
COLAB_SCOPE = "https://www.googleapis.com/auth/colaboratory"


def _force_restricted_refresh(auth_module) -> None:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_path = Path(auth_module.TOKEN_CONFIG_PATH).expanduser()
    os.chmod(token_path.parent, 0o700)
    os.chmod(token_path, 0o600)
    payload = json.loads(token_path.read_text(encoding="utf-8"))
    credentials = Credentials.from_authorized_user_info(
        payload, scopes=list(APPROVED_SCOPES)
    )
    credentials.refresh(Request())

    # Google returns the actual grant.  Cached metadata alone only describes
    # requested scopes, so verify the refreshed access token independently.
    endpoint = "https://oauth2.googleapis.com/tokeninfo?" + urlencode(
        {"access_token": credentials.token}
    )
    with urlopen(endpoint, timeout=25) as response:
        actual = set(json.load(response).get("scope", "").split())
    if COLAB_SCOPE not in actual or not actual <= APPROVED_TOKEN_SCOPES:
        raise RuntimeError("Refreshed token exceeds the approved scope subset")

    refreshed = json.loads(credentials.to_json())
    refreshed["scopes"] = list(APPROVED_SCOPES)
    temporary = token_path.with_suffix(".restricted-refresh.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(refreshed, output)
        os.replace(temporary, token_path)
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(token_path, 0o600)
    print(json.dumps({
        "colab_auth_refresh": "complete",
        "actual_scopes": sorted(actual),
        "persisted_scopes_approved_only": True,
        "token_mode": "0o600",
    }), flush=True)


def main() -> None:
    from colab_cli import auth

    os.umask(0o077)
    # In PyPI 0.7.5 all OAuth creation, cache loading and refresh scope requests
    # reference this module constant.  This is process-local and only reduces
    # permissions; Google's server-side authorization remains unchanged.
    auth.PUBLIC_SCOPES = list(APPROVED_SCOPES)
    if len(sys.argv) > 1 and sys.argv[1] == "--refresh-auth":
        del sys.argv[1]
        try:
            _force_restricted_refresh(auth)
        except Exception as error:
            # Never serialize exceptions that may include an access-token URL.
            print(json.dumps({
                "colab_auth_refresh": "failed",
                "error_type": type(error).__name__,
            }), file=sys.stderr, flush=True)
            raise SystemExit(1) from None

    from colab_cli.cli import main as official_main
    official_main()


if __name__ == "__main__":
    main()
