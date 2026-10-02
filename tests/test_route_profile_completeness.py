"""Consumer-boundary gate: the public OpenAI surface registered in create_app must
equal the endpoint-profile (operations + declared extensions) in BOTH directions.
A public route with no profile row, or a profile row with no public route, fails CI.
The POST /v1/embeddings and POST /moderations proofs are the motivating cases.
The planted example is a route that does not exist: /v1/embeddings became
real on 2026-09-21 and a synthetic 'unprofiled route' has to stay synthetic.
"""
import re, pathlib
from fastapi.routing import APIRoute
from chord.server import Deps, create_app
from chord.config import Settings
from qa.conformance.profile import load_profile
import test_progress as tp

REPO = pathlib.Path(__file__).resolve().parents[1]
PROFILE = load_profile(REPO / "qa" / "conformance" / "endpoint-profile.json")
FASTAPI_BUILTINS = {"GET /openapi.json","GET /docs","GET /redoc","GET /docs/oauth2-redirect","HEAD /openapi.json"}

def _norm(method, path):
    # A spec alias like "POST /responses?beta=true" is served by the same route:
    # the query string is not part of a route's path (#146 beta aliases).
    path = path.split("?", 1)[0]
    p = path[3:] if path.startswith("/v1") else path
    return f"{method} {re.sub(r'\{[^}]+\}', '{}', p)}"

def _profiled():  # operations + declared extensions
    keys = set(PROFILE["operations"]) | set(PROFILE.get("extensions", {}))
    return {_norm(*k.split(" ", 1)) for k in keys}

def _app_surface(tmp_path):
    app = create_app(Deps(Settings(data_dir=tmp_path),
                          upstream=tp.FakeUpstream(), model=lambda n: tp.FixedRouter()))
    out = set()
    for r in app.routes:
        if isinstance(r, APIRoute):
            for m in (r.methods or set()):
                if m in ("GET","POST","PUT","PATCH","DELETE"):
                    n = _norm(m, r.path)
                    if n not in FASTAPI_BUILTINS: out.add(n)
    return out

def test_route_profile_equality_both_directions(tmp_path):
    app, prof = _app_surface(tmp_path), _profiled()
    assert not (app - prof), f"public routes with no profile/extension row: {sorted(app - prof)}"
    assert not (prof - app), f"profile/extension rows with no public route: {sorted(prof - app)}"

def test_planted_unprofiled_route_is_detected(tmp_path):
    assert (_app_surface(tmp_path) | {"POST /nonexistent"}) - _profiled() == {"POST /nonexistent"}

def test_planted_profile_only_row_is_detected(tmp_path):
    """The plant has to be a route that will never exist. It was `POST /moderations`
    until 2026-09-18, when that became a real served no-op and the plant quietly
    stopped being a plant — a control that passes because it is no longer testing
    anything."""
    fake = "POST /a-route-that-will-never-exist"
    assert (_profiled() | {fake}) - _app_surface(tmp_path) == {fake}


def test_a_query_alias_row_needs_its_base_route(tmp_path):
    """Stripping ?beta=true must not let an alias stand in for a route that doesn't exist."""
    assert _norm("POST", "/nonexistent?beta=true") == "POST /nonexistent"
    assert "POST /nonexistent" not in _app_surface(tmp_path)
