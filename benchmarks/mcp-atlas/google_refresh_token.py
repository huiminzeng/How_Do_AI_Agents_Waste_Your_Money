"""Obtain a new GOOGLE_REFRESH_TOKEN for the google-workspace MCP server.

The server (@geobio/google-workspace-server) authenticates with an OAuth refresh token
for the scopes calendar, gmail.modify and gmail.send. Google expires or revokes such
tokens (an expired one answers `invalid_grant` - "Token has been expired or
revoked"), and the only way to get a new one is a browser sign-in by the
Google account whose calendar and mailbox the tasks refer to.

No extra dependencies. Run from mcp-atlas/:

    python3 scripts/google_refresh_token.py

1. It prints a URL. Open it in a browser and sign in with the benchmark's Google
   account, accept the three permissions.
2. Google redirects the browser to http://localhost:4100/code?code=... - that page will not
   load if the browser is on another machine; that is fine. Copy the `code=` value out of
   the address bar and paste it here.
3. The script exchanges the code and prints the refresh token. Put it in mcp-atlas/.env as
   GOOGLE_REFRESH_TOKEN and restart the sandbox container (the value is read at start).

The OAuth client in .env must permit the loopback redirect (Desktop-type clients do; a
Web-type client needs the redirect URI below in its authorised redirect URIs).
"""
import json
import os
import sys
import urllib.parse
import urllib.request

SCOPES = (
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
)
# The redirect URI registered on the benchmark's OAuth client. It is the one
# @geobio/google-workspace-server's README tells you to register ("Authorized redirect
# URIs ... http://localhost:4100/code"); http://localhost:8765 was refused with
# redirect_uri_mismatch, so the client is Web-type and only knows this one.
REDIRECT = "http://localhost:4100/code"



def _extract_code(raw: str) -> str:
    """The authorisation code, from a bare code or from the whole redirected URL.

    Pasting the wrong thing is the normal failure here, twice now: the redirect lands on
    a page the browser cannot load, so it is not obvious that the address bar holds the
    answer, and the authorise URL shown earlier also contains a `redirect_uri=` that looks
    like something worth copying. Accept either form and pull the code out.
    """
    raw = raw.strip().strip('"').strip("'")
    if "code=" in raw:
        raw = raw.split("code=", 1)[1].split("&", 1)[0]
    elif raw.startswith("http"):
        return ""          # a URL with no code in it: the wrong page was copied
    code = urllib.parse.unquote(raw).strip()
    # Google's authorisation codes look like "4/0A...". Anything else - a redirect_uri, the
    # consent-warning URL, the authorise URL - only earns a "Malformed auth code" from the
    # token endpoint, which does not tell the operator what they actually did wrong.
    return code if code.startswith("4/") else ""


def _env() -> dict:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for line in open(os.path.join(here, ".env")):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            out[k] = v
    return out


def main() -> int:
    env = _env()
    cid, secret = env.get("GOOGLE_CLIENT_ID"), env.get("GOOGLE_CLIENT_SECRET")
    if not cid or not secret:
        print("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET missing in mcp-atlas/.env", file=sys.stderr)
        return 1
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
        "client_id": cid, "redirect_uri": REDIRECT, "response_type": "code",
        "scope": " ".join(SCOPES), "access_type": "offline", "prompt": "consent",
    })
    print("\nOpen this URL, sign in with the benchmark's Google account, accept:\n")
    print(url)
    raw = input("\nPaste the WHOLE address bar from the page that failed to load\n"
                "(it starts http://localhost:4100/code?code=...), or just the code: ").strip()
    code = _extract_code(raw)
    if not code:
        print("That is not an authorisation code.\n"
              "  - If you pasted an accounts.google.com/... URL you are still on the consent or\n"
              "    warning screen. Click Advanced, then 'Go to ... (unsafe)', then approve.\n"
              "  - The page you want is the one that fails to load with ERR_CONNECTION_REFUSED.\n"
              "    Its ADDRESS BAR reads http://localhost:4100/code?code=4/0A...  Copy that.\n"
              "  - A code starts with '4/'.", file=sys.stderr)
        return 1
    data = urllib.parse.urlencode({
        "code": code, "client_id": cid, "client_secret": secret,
        "redirect_uri": REDIRECT, "grant_type": "authorization_code",
    }).encode()
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request("https://oauth2.googleapis.com/token", data=data), timeout=30)
    except urllib.error.HTTPError as e:
        print("token exchange failed:", e.code, e.read().decode()[:400], file=sys.stderr)
        return 1
    tok = json.loads(resp.read())
    rt = tok.get("refresh_token")
    if not rt:
        print("no refresh_token in the response (was prompt=consent honoured?):",
              {k: v for k, v in tok.items() if k != "access_token"}, file=sys.stderr)
        return 1
    print("\nGOOGLE_REFRESH_TOKEN=" + rt)
    print("\nPut that line in mcp-atlas/.env, then restart the sandbox container.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
