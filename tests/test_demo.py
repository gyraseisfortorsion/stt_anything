"""Demo jobs and HTTP contracts, without model downloads or a microphone."""

import json
import runpy
import sys
import threading
import time
from http.client import HTTPConnection
from unittest.mock import Mock

import numpy as np
import pytest
from conftest import FakeEncoder, FakeSource

from demo import server, service
from demo.service import (
    APIError,
    DemoApp,
    ProgressEncoder,
    identifier,
    terms_from_request,
    waveform,
)
from stt_anything.models import DetectionResult, Hit


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(service.components, "encoder", lambda *a: FakeEncoder())
    monkeypatch.setattr(service.components, "source", lambda *a: FakeSource())
    instance = DemoApp(tmp_path)
    yield instance
    instance.close()


def finished(app, submitted):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = app.job(submitted["job_id"])
        if job["state"] in ("complete", "error"):
            return job
        time.sleep(0.005)
    pytest.fail("Demo job did not finish")


def payload(**values):
    return {
        "language": "ru",
        "name": "Test",
        "encoder": "mfcc",
        "terms": [{"display": "омепразол"}],
        **values,
    }


def test_dictionary_and_progress():
    terms = terms_from_request(
        payload(terms=[{"display": "омепразол", "aliases": ["O"], "spoken_forms": ["M"]}])
    )
    assert terms[0].forms == ("омепразол", "O", "M")
    updates = []
    encoder = ProgressEncoder(FakeEncoder(), updates.append)
    assert len(encoder.encode(np.ones(10)).values) == 6
    encoder.encode_term("x", "ru", FakeSource())
    assert updates == ["x"]
    assert waveform(np.array([-0.3, 0.9, 0, 0.1], dtype=np.float32), 2) == [[-0.3, 0.9], [0, 0.1]]
    assert len(waveform(np.ones(2, dtype=np.float32))) == 2
    assert identifier("a" * 32) == "a" * 32
    for value in ("", "g" * 32):
        with pytest.raises(APIError):
            identifier(value)


@pytest.mark.parametrize(
    "data",
    [
        {},
        payload(language="fr"),
        payload(terms="x"),
        payload(terms=[]),
        payload(terms=[{}] * 101),
        payload(terms=["x"]),
        payload(terms=[{"display": "x", "language": "en"}]),
        payload(terms=[{"display": "x"}, {"display": "X"}]),
        payload(terms=[{"display": "x" * 151}]),
        payload(terms=[{"display": "x", "aliases": [str(i) for i in range(301)]}]),
    ],
)
def test_invalid_dictionary(data):
    with pytest.raises(ValueError):
        terms_from_request(data)


@pytest.mark.parametrize(
    "values",
    [
        {"name": ""},
        {"name": 3},
        {"encoder": "unknown"},
        {"top_k": 1},
        {"strong_threshold": 0.1, "possible_threshold": 0.5},
    ],
)
def test_invalid_index(app, values):
    with pytest.raises(ValueError):
        app.create_index(payload(**values))


def test_indexes_and_cache(app, tmp_path, monkeypatch):
    job = finished(app, app.create_index(payload()))
    assert job["state"] == "complete"
    info = job["result"]
    assert info["templates"] == 1 and app.index_info(info["id"]) == json.loads(json.dumps(info))
    assert job["completed"] == 1 and job["progress"] == 100
    assert app.encoder("mfcc", "ru", True) is app.encoder("mfcc", "ru", True)
    assert app.encoder("mfcc", "en", True) is not app.encoder("mfcc", "ru", True)
    for content in (
        "{}",
        "not JSON",
        '{"id":"bad"}',
        json.dumps({"id": "b" * 32, "created_at": 0}),
    ):
        (tmp_path / "broken.json").write_text(content)
        assert len(app.indexes()) == 1
    with pytest.raises(APIError):
        app.index_info("b" * 32)
    with pytest.raises(APIError):
        app.job("b" * 32)
    with pytest.raises(APIError):
        app.audio(job["id"])
    monkeypatch.setattr(service, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(service.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(service.shutil, "which", lambda name: "espeak-ng")
    assert app.config()["encoders"][0]["installed"]
    monkeypatch.setattr(service.importlib.util, "find_spec", lambda name: None)
    assert not app.config()["encoders"][0]["installed"]
    monkeypatch.setattr(service, "cache_root", lambda: tmp_path)
    other = DemoApp()
    other.close()


def test_jobs_errors(app):
    def fail(job_id):
        raise RuntimeError("failed model")

    assert finished(app, app.submit(fail))["message"] == "failed model"
    first = finished(app, app.create_index(payload(encoder="phoneme", allow_downloads=True)))
    assert first["result"]["lookup"] == "phoneme"


@pytest.mark.parametrize("lookup", ["dtw", "phoneme", "characters"])
def test_analyze(app, monkeypatch, lookup):
    info = finished(
        app,
        app.create_index(
            payload(encoder={"phoneme": "phoneme", "characters": "russian-ctc"}.get(lookup, "mfcc"))
        ),
    )["result"]
    monkeypatch.setattr(service, "read_audio", lambda path: np.ones(16000, dtype=np.float32) * 0.1)
    hit = Hit("ru-0", "омепразол", 0.1, 0.5, 0.8, "strong")
    detector = Mock()
    detector.detect.return_value = DetectionResult("ru", 1, "fake", lookup, (hit,))
    monkeypatch.setattr(service, "KeywordDetector", lambda *a: detector)
    job = finished(app, app.analyze(info["id"], b"fake-audio"))
    assert job["state"] == "complete"
    assert job["result"]["hits"][0]["term"] == "омепразол"
    assert len(job["result"]["peaks"]) == 1200
    assert app.audio(job["id"]).startswith(b"RIFF")
    with pytest.raises(APIError):
        app.analyze(info["id"], b"")
    monkeypatch.setattr(service, "MAX_UPLOAD", 1)
    with pytest.raises(APIError):
        app.analyze(info["id"], b"aa")
    monkeypatch.setattr(service, "MAX_DURATION", 0)
    failed = finished(app, app.analyze(info["id"], b"a"))
    assert failed["state"] == "error"
    with pytest.raises(APIError):
        app.audio(failed["id"])


@pytest.fixture
def http(app):
    instance = server.DemoServer(("127.0.0.1", 0), app)
    worker = threading.Thread(target=instance.serve_forever)
    worker.start()

    def request(method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", instance.server_port)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    yield request, instance, app
    instance.shutdown()
    worker.join()
    instance.server_close()


def test_http_routes(http, monkeypatch):
    request, instance, app = http
    for path in server.ASSETS:
        status, body, headers = request("GET", path)
        assert status == 200 and body and headers["Cache-Control"] == "no-store"
    assert request("GET", "/api/config")[0] == 200
    assert request("GET", "/api/indexes")[0] == 200
    assert request("GET", "/missing")[0] == 404
    assert request("GET", "/api/jobs/bad")[0] == 404
    assert request("GET", "/api/audio/bad")[0] == 404
    assert request("GET", "/", headers={"Host": "attacker.invalid"})[0] == 403
    assert request("GET", "/", headers={"Origin": "http://attacker.invalid"})[0] == 403
    assert (
        request("GET", "/", headers={"Origin": f"https://localhost:{instance.server_port}"})[0]
        == 403
    )
    assert (
        request("GET", "/", headers={"Origin": f"http://localhost:{instance.server_port}"})[0]
        == 200
    )
    assert request("POST", "/api/indexes")[0] == 413
    assert request("POST", "/api/indexes", "[]")[0] == 400
    assert request("POST", "/api/indexes", "invalid")[0] == 400
    assert request("POST", "/missing", "{}")[0] == 404
    assert request("POST", "/api/runs", "audio")[0] == 404
    status, body, _ = request("POST", "/api/indexes", json.dumps(payload()))
    assert status == 202
    job = finished(app, json.loads(body))
    assert request("GET", f"/api/jobs/{job['id']}")[0] == 200
    monkeypatch.setattr(service, "read_audio", lambda path: np.ones(16000, dtype=np.float32) * 0.1)
    status, body, _ = request("POST", f"/api/runs?index_id={job['result']['id']}", b"audio")
    assert status == 202
    run = finished(app, json.loads(body))
    assert run["state"] == "complete"
    assert request("GET", f"/api/audio/{run['id']}")[2]["Content-Type"] == "audio/wav"
    monkeypatch.setattr(server, "MAX_UPLOAD", 1)
    assert request("POST", "/api/indexes", "{}")[0] == 413


def test_main(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["demo", "--port", "0", "--data-dir", str(tmp_path)])
    monkeypatch.setattr(server.DemoServer, "serve_forever", Mock(side_effect=KeyboardInterrupt))
    server.main()
    monkeypatch.setattr(server.DemoServer, "serve_forever", Mock())
    server.main()
    monkeypatch.setattr(
        server.ThreadingHTTPServer, "serve_forever", Mock(side_effect=KeyboardInterrupt)
    )
    with pytest.warns(RuntimeWarning):
        runpy.run_module("demo.server", run_name="__main__")


def test_english_dictionary_rejects_russian_pronunciations():
    assert (
        terms_from_request(payload(language="en", terms=[{"display": "aspirin"}]))[0].language
        == "en"
    )
    with pytest.raises(APIError, match="Russian index"):
        terms_from_request(payload(language="en", terms=[{"display": "зиртек"}]))
    with pytest.raises(APIError, match="Russian index"):
        terms_from_request(
            payload(language="en", terms=[{"display": "brand", "spoken_forms": ["Бренд"]}])
        )


def test_lookup_diagnostics(index):
    from conftest import FakeLookup

    from demo.diagnostics import ObservedLookup, diagnostics

    lookup = ObservedLookup(FakeLookup())
    query = FakeEncoder().encode(np.ones(100, dtype=np.float32))
    candidates = lookup.search(index, query)
    assert lookup.name == "fake-lookup"
    report = diagnostics(np.ones(100, dtype=np.float32) * 0.1, candidates, 0.5)
    assert report["candidates_above_threshold"] == 1
    assert report["best_candidate_score"] == 0.9 and not report["quiet_audio"]
    assert diagnostics(np.zeros(100, dtype=np.float32), [], 0.5)["quiet_audio"]
    assert (
        diagnostics(np.ones(100, dtype=np.float32), candidates, 1)["candidates_above_threshold"]
        == 0
    )


def test_russian_demo_language_restriction(app):
    with pytest.raises(APIError, match="Russian dictionary"):
        app.create_index(
            payload(language="en", encoder="russian-ctc", terms=[{"display": "aspirin"}])
        )
