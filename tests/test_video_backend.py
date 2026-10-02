"""The ComfyUI client and the operator video workflow backend.

The `/history` and `/queue` shapes asserted here follow real ComfyUI responses,
not a reading of what an endpoint probably returns.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from chord import comfy, video


def _transport(handler):
    return httpx.AsyncClient(base_url="http://comfy.invalid",
                             transport=httpx.MockTransport(handler))


def _client(handler) -> comfy.ComfyClient:
    return comfy.ComfyClient("http://comfy.invalid", client=_transport(handler))


VIDEO_WORKFLOW = video.VideoWorkflow(
    graph={
        "input": {"class_type": "VideoSampler", "inputs": {
            "prompt": "sample", "width": 1, "height": 1, "seconds": 1, "seed": 0,
        }},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["input", 0]}},
    },
    output_node_id="save",
    bindings={name: ("input", field) for name, field in {
        "prompt": "prompt", "width": "width", "height": "height",
        "duration": "seconds", "seed": "seed",
    }.items()},
)


# ---------------------------------------------------------------- the client

@pytest.mark.asyncio
async def test_upload_returns_comfyuis_name_not_ours():
    # ComfyUI renames on collision. A graph injected with the name we SENT would
    # then load a different caller's picture.
    def handler(request):
        return httpx.Response(200, json={"name": "ref (2).png", "subfolder": ""})

    assert await _client(handler).upload_image(b"bytes", "ref.png") == "ref (2).png"


@pytest.mark.asyncio
async def test_upload_preserves_a_subfolder():
    def handler(request):
        return httpx.Response(200, json={"name": "ref.png", "subfolder": "clips"})

    assert await _client(handler).upload_image(b"b", "ref.png") == "clips/ref.png"


@pytest.mark.asyncio
async def test_submit_treats_a_200_without_prompt_id_as_rejection():
    # ComfyUI reports graph validation failures in the body as often as by
    # status; trusting the status alone loses the error entirely.
    def handler(request):
        return httpx.Response(200, json={"error": {"type": "prompt_outputs_failed_validation"}})

    with pytest.raises(comfy.ComfyError):
        await _client(handler).submit({"1": {"class_type": "X"}})


@pytest.mark.asyncio
async def test_submit_refuses_an_empty_graph_before_the_network():
    with pytest.raises(ValueError):
        await _client(lambda r: httpx.Response(200, json={})).submit({})


@pytest.mark.asyncio
async def test_poll_reports_queued_running_and_completed():
    state = {"history": {}, "running": False}

    def handler(request):
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json=state["history"])
        return httpx.Response(200, json={
            "queue_running": [[0, "p1"]] if state["running"] else []})

    client = _client(handler)
    assert await client.poll("p1") == (comfy.QUEUED, 0.0)

    state["running"] = True
    assert (await client.poll("p1"))[0] == comfy.RUNNING

    state["history"] = {"p1": {"status": {"completed": True, "status_str": "success"},
                               "outputs": {}}}
    assert await client.poll("p1") == (comfy.COMPLETED, 1.0)


@pytest.mark.asyncio
async def test_poll_reports_failure_for_an_errored_prompt():
    def handler(request):
        return httpx.Response(200, json={"p1": {"status": {"completed": True,
                                                           "status_str": "error"}}})

    assert (await _client(handler).poll("p1"))[0] == comfy.FAILED


@pytest.mark.asyncio
async def test_fetch_finds_an_artifact_by_shape_not_by_saver_node():
    # SaveImage writes `images`, SaveVideo writes `videos`, a custom node writes
    # whatever it likes. Matching the record shape means a new saver works on
    # arrival instead of returning "completed but saved nothing".
    for key in ("images", "videos", "some_future_key"):
        def handler(request, key=key):
            if request.url.path.startswith("/history/"):
                return httpx.Response(200, json={"p1": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {"92": {key: [{"filename": "out.mp4",
                                              "subfolder": "video",
                                              "type": "output"}]}}}})
            return httpx.Response(200, content=b"\x00mp4", headers={
                "content-type": "application/octet-stream"})

        data, content_type = await _client(handler).fetch("p1")
        assert data == b"\x00mp4"
        assert content_type == "video/mp4"


@pytest.mark.asyncio
async def test_fetch_names_the_failing_node_rather_than_saying_failed():
    def handler(request):
        return httpx.Response(200, json={"p1": {"status": {
            "completed": True, "status_str": "error",
            "messages": [["execution_error", {"node_type": "UNETLoader",
                                              "exception_message": "no such model"}]]}}})

    with pytest.raises(comfy.ComfyError, match="UNETLoader"):
        await _client(handler).fetch("p1")


@pytest.mark.asyncio
async def test_fetch_prefers_the_mp4_when_a_preview_is_listed_first():
    def handler(request):
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json={"p1": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {"92": {
                    "images": [{"filename": "preview.png", "subfolder": "", "type": "output"}],
                    "videos": [{"filename": "out.mp4", "subfolder": "video", "type": "output"}],
                }}}})
        name = request.url.params.get("filename")
        body = b"\x00mp4" if name == "out.mp4" else b"\x89PNG"
        return httpx.Response(200, content=body, headers={"content-type": "application/octet-stream"})

    data, content_type = await _client(handler).fetch("p1")
    assert data == b"\x00mp4"
    assert content_type == "video/mp4"


@pytest.mark.asyncio
async def test_fetch_uses_only_the_operator_bound_video_output():
    def handler(request):
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json={"p1": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {
                    "other": {"videos": [{"filename": "wrong.mp4", "type": "output"}]},
                    "save": {"videos": [{"filename": "right.mp4", "type": "output"}]},
                }}})
        return httpx.Response(200, content=request.url.params["filename"].encode())

    client = _client(handler)
    assert await client.fetch("p1", "save") == (b"right.mp4", "video/mp4")
    with pytest.raises(comfy.ComfyError, match="output node"):
        await client.fetch("p1", "missing")


@pytest.mark.asyncio
async def test_a_render_timeout_is_wall_clock_and_interrupts_the_job():
    seen: list[tuple[str, str]] = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json={})
        if request.url.path in ("/queue", "/interrupt"):
            # p1 is the prompt executing: the stop is ours to send (B20).
            return httpx.Response(200, json={"queue_running": [[0, "p1", {}, {}, []]],
                                             "queue_pending": []})
        return httpx.Response(404)

    with pytest.raises(video.VideoError, match="timed out"):
        await video.render(
            _client(handler), VIDEO_WORKFLOW,
            {"prompt": "clouds", "width": 1280, "height": 720, "seconds": 8},
            poll_interval=0.01, timeout=0.05,
        )
    assert ("POST", "/interrupt") in seen


@pytest.mark.asyncio
async def test_fetch_refuses_a_completed_render_that_did_not_succeed():
    """Poll calls this FAILED and expects fetch to raise. A partial SaveVideo
    file must not come back as the artifact."""
    def handler(request):
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json={"p1": {
                "status": {"completed": True, "status_str": "interrupted"},
                "outputs": {"92": {"videos": [{"filename": "partial.mp4", "subfolder": "", "type": "output"}]}}}})
        return httpx.Response(200, content=b"\x00mp4")

    with pytest.raises(comfy.ComfyError, match="failed"):
        await _client(handler).fetch("p1")


@pytest.mark.asyncio
async def test_fetch_distinguishes_unfinished_from_produced_nothing():
    def unknown(request):
        return httpx.Response(200, json={})

    with pytest.raises(comfy.ComfyError, match="unfinished or unknown"):
        await _client(unknown).fetch("p1")

    def empty(request):
        return httpx.Response(200, json={"p1": {
            "status": {"completed": True, "status_str": "success"}, "outputs": {}}})

    with pytest.raises(comfy.ComfyError, match="saved no artifact"):
        await _client(empty).fetch("p1")


@pytest.mark.asyncio
async def test_an_http_error_carries_its_status():
    def handler(request):
        return httpx.Response(503, text="upstream busy")

    with pytest.raises(comfy.ComfyError) as caught:
        await _client(handler).submit({"1": {}})
    assert caught.value.status == 503


# ---------------------------------------------------------------- the backend

def test_backend_is_none_without_a_configured_comfyui():
    class Settings:
        comfy_base_url = ""

    assert video.VideoBackend.from_settings(Settings()) is None


def test_backend_is_built_when_comfyui_is_configured():
    class WorkflowConfig:
        def load(self):
            return VIDEO_WORKFLOW

    class Settings:
        comfy_base_url = "http://comfy.invalid"
        video_timeout_s = 60.0
        video_workflows = {"t2v": WorkflowConfig(), "r2v": WorkflowConfig()}

    backend = video.VideoBackend.from_settings(Settings())
    assert isinstance(backend, video.VideoBackend)
    assert backend.supports("t2v") and backend.supports("r2v")


@pytest.mark.asyncio
async def test_a_queued_render_bounds_its_wait_instead_of_hanging():
    """The lock models the one GPU, and waiting on it used to be unbounded: an
    inline image edit hung its HTTP connection for the whole video queue, and a
    queued video pinned its reference closure for the same hours (review
    2026-09-22). The wait is bounded by one render's timeout, and the timeout
    must leave the lock exactly as it found it -- held by the other render."""
    backend = video.VideoBackend(_client(lambda request: httpx.Response(200, json={})),
                                 {"t2v": VIDEO_WORKFLOW}, timeout=0.05)
    lock = await backend._lock()
    await lock.acquire()                       # the GPU is "busy" with another render
    with pytest.raises(TimeoutError):
        await backend.render("t2v", {})
    assert lock.locked()                       # not leaked, not stolen
    lock.release()


@pytest.mark.asyncio
async def test_no_history_call_can_disable_its_timeout():
    """httpx reads an explicit timeout=None as NO timeouts at all, not the
    client default. A stalled /history call happens while the caller holds the
    one-render lock, so an unbounded one wedges every later edit, variation
    and video until process restart (review 2026-09-22, #4)."""
    seen: list = []

    class Recording:
        async def get(self, url, **kw):
            seen.append((url, kw.get("timeout", "ABSENT")))
            return httpx.Response(200, json={})

        async def post(self, url, **kw):
            seen.append((url, kw.get("timeout", "ABSENT")))
            return httpx.Response(200, json={})

    client = comfy.ComfyClient("http://comfy.invalid", client=Recording())
    with pytest.raises(comfy.ComfyError):        # empty history: unfinished
        await client.fetch("p1")
    with pytest.raises(comfy.ComfyError):
        await client.fetch_pngs("p1")
    history_calls = [t for u, t in seen if "/history/" in u]
    assert history_calls, "fetch must consult history"
    assert all(isinstance(t, (int, float)) and t > 0 for t in history_calls), seen


class _InterruptibleFake:
    """Submits, then runs forever until cancelled, recording the interrupt."""

    def __init__(self) -> None:
        self.submitted: list = []
        self.interrupted: list = []

    async def upload_image(self, data, filename):
        return filename

    async def submit(self, graph):
        self.submitted.append(len(self.submitted) + 1)
        return "prompt-1"

    async def poll(self, prompt_id):
        return comfy.RUNNING, 0.5

    async def fetch(self, prompt_id, output_node_id=None):
        raise AssertionError("a cancelled render must not fetch")

    async def fetch_pngs(self, prompt_id):
        raise AssertionError("a cancelled render must not fetch")

    async def interrupt(self, prompt_id):
        self.interrupted.append(prompt_id)


@pytest.mark.asyncio
async def test_a_cancelled_video_render_interrupts_the_job_it_submitted():
    """The probe (batch-1 review, the required fix): the cancel used to end
    only chord's WAIT -- the submitted ComfyUI job kept rendering, the next
    job submitted while the orphan held the GPU, and a create/delete loop grew
    ComfyUI's own queue past every admission bound. Under the one-render lock
    the global interrupt can only hit OUR prompt -- the same guarantee the
    timeout path already relied on -- so the CancelledError path interrupts
    before the lock releases."""
    fake = _InterruptibleFake()
    backend = video.VideoBackend(fake, {"t2v": VIDEO_WORKFLOW}, timeout=60.0)
    task = asyncio.create_task(backend.render(
        "t2v", {"prompt": "waves", "width": 720, "height": 1280, "seconds": 4, "seed": 1}))
    await asyncio.sleep(0.05)                    # let it submit and enter the poll loop
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.submitted == [1]
    assert fake.interrupted == ["prompt-1"]
    lock = await backend._lock()
    assert not lock.locked()                     # released: no orphan holds the GPU



# ------------------------------------------------- stopping only our prompt (B20)

def _queue_handler(running: list[str], pending: list[str], seen: list):
    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(200, json={
                "queue_running": [[0, pid, {}, {}, []] for pid in running],
                "queue_pending": [[i + 1, pid, {}, {}, []] for i, pid in enumerate(pending)],
            })
        if request.method == "POST" and request.url.path in ("/queue", "/interrupt"):
            return httpx.Response(200, json={})
        return httpx.Response(404)
    return handler


@pytest.mark.asyncio
async def test_interrupt_never_stops_a_prompt_that_is_not_ours():
    """Review 2026-09-24 B20: ComfyUI's POST /interrupt stops whatever is
    executing, and other clients may share the ComfyUI host. A Chord timeout
    or DELETE whose prompt is not the one running
    must not send it: that would kill someone else's render."""
    seen: list = []
    await _client(_queue_handler(["other-job"], [], seen)).interrupt("p1")
    assert not [s for s in seen if s[1] == "/interrupt"], seen


@pytest.mark.asyncio
async def test_a_pending_prompt_of_ours_is_deleted_from_the_queue_not_interrupted():
    """Review 2026-09-24 B20: a Chord prompt still waiting behind someone
    else's render is removed from the queue by id; the running one is left
    alone."""
    seen: list = []
    await _client(_queue_handler(["other-job"], ["p1"], seen)).interrupt("p1")
    assert ("POST", "/queue", {"delete": ["p1"]}) in seen
    assert not [s for s in seen if s[1] == "/interrupt"], seen


@pytest.mark.asyncio
async def test_a_running_prompt_of_ours_is_interrupted_by_id():
    """Review 2026-09-24 B20: only when Chord's prompt is the one executing is
    /interrupt sent, and it names the prompt, so a ComfyUI that supports a
    targeted interrupt refuses to stop anything else if the running prompt
    changed between the read and the stop."""
    seen: list = []
    await _client(_queue_handler(["p1"], ["next-job"], seen)).interrupt("p1")
    assert ("POST", "/interrupt", {"prompt_id": "p1"}) in seen
    assert not [s for s in seen if s[1] == "/queue" and s[0] == "POST"
                and s[2] != {"delete": ["p1"]}], seen


@pytest.mark.asyncio
async def test_an_unreadable_queue_never_falls_back_to_the_global_interrupt():
    """Review 2026-09-24 B20: when the queue cannot be read, whose prompt is
    running is unknown, and unknown is not ours."""
    seen: list = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={})

    await _client(handler).interrupt("p1")
    assert ("POST", "/interrupt") not in seen, seen


@pytest.mark.asyncio
async def test_interrupt_reports_whether_the_stop_is_settled():
    """Copilot on #336: a startup cancel that could not reach ComfyUI was
    indistinguishable from one that worked, because every failure here is
    swallowed. The caller now learns which: settled when the queue was read
    and every stop it needed was accepted, or the prompt was not there."""
    seen: list = []
    assert await _client(_queue_handler(["p1"], [], seen)).interrupt("p1") is True
    assert await _client(_queue_handler([], [], seen)).interrupt("p1") is True

    def unreadable(request):
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={})
    assert await _client(unreadable).interrupt("p1") is False

    def refusing(request):
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [[0, "p1", {}, {}, []]], "queue_pending": []})
        return httpx.Response(500, text="no")
    assert await _client(refusing).interrupt("p1") is False


@pytest.mark.asyncio
async def test_a_failing_submit_callback_still_stops_the_prompt_it_submitted():
    """Copilot on #336: on_submit runs after ComfyUI accepted the prompt. When
    it raised anything but a cancel (a disk error keeping the id), only
    CancelledError was handled, so the error propagated, the row went failed,
    and the prompt rendered on as an orphan nothing knew about."""
    fake = _SettlingFake(True)
    backend = video.VideoBackend(fake, {"t2v": VIDEO_WORKFLOW}, timeout=60.0)
    settled: list = []

    def failing(prompt_id):
        raise OSError("disk full")

    with pytest.raises(OSError):
        await backend.render("t2v", {"prompt": "waves", "width": 720, "height": 1280,
                                     "seconds": 4, "seed": 1}, on_submit=failing,
                             on_settled=settled.append)
    assert fake.interrupted == ["prompt-1"]
    # The callback may have recorded the id before raising: a settled stop
    # clears it (Copilot on #336).
    assert settled == ["prompt-1"]
    lock = await backend._lock()
    assert not lock.locked()


@pytest.mark.asyncio
async def test_a_pending_prompt_promoted_while_being_deleted_is_still_stopped():
    """Copilot on #336: the queue read and the stop are not atomic. Our prompt
    was pending when read, ComfyUI promoted it before the delete landed, the
    delete was a no-op, and interrupt() still reported the stop settled while
    the prompt rendered. The queue is read again after a delete."""
    seen: list = []
    reads = {"n": 0}

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path == "/queue":
            reads["n"] += 1
            if reads["n"] == 1:
                return httpx.Response(200, json={"queue_running": [[0, "other-job", {}, {}, []]],
                                                 "queue_pending": [[1, "p1", {}, {}, []]]})
            return httpx.Response(200, json={"queue_running": [[0, "p1", {}, {}, []]], "queue_pending": []})
        return httpx.Response(200, json={})

    assert await _client(handler).interrupt("p1") is True
    assert ("POST", "/interrupt", {"prompt_id": "p1"}) in seen, seen


class _SlowSubmitFake(_InterruptibleFake):
    """ComfyUI accepts the prompt, but the answer arrives after the cancel."""

    def __init__(self) -> None:
        super().__init__()
        self.accepted = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, graph):
        self.accepted.set()
        await self.release.wait()
        return "prompt-1"


@pytest.mark.asyncio
async def test_a_cancel_while_the_submit_is_in_flight_still_stops_the_prompt():
    """Copilot on #336: cleanup began only after submit() returned. A DELETE
    or shutdown landing after ComfyUI accepted /prompt but before its answer
    arrived left with no id: nothing stopped the prompt and nothing recorded
    it. The submit is shielded, so its id is recorded and stopped."""
    fake = _SlowSubmitFake()
    recorded: list = []
    backend = video.VideoBackend(fake, {"t2v": VIDEO_WORKFLOW}, timeout=60.0)
    task = asyncio.create_task(backend.render(
        "t2v", {"prompt": "waves", "width": 720, "height": 1280, "seconds": 4, "seed": 1},
        on_submit=recorded.append))
    await fake.accepted.wait()
    task.cancel()
    await asyncio.sleep(0)
    fake.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert recorded == ["prompt-1"]
    assert fake.interrupted == ["prompt-1"]


class _SettlingFake(_InterruptibleFake):
    def __init__(self, stop_settles: bool) -> None:
        super().__init__()
        self.stop_settles = stop_settles

    async def interrupt(self, prompt_id):
        self.interrupted.append(prompt_id)
        return self.stop_settles


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_settles", [True, False])
async def test_a_stopped_render_reports_settled_only_when_the_stop_settled(stop_settles):
    """Copilot on #336: the submitted-prompt record was cleared only when a
    render returned. A timeout or a cancel whose targeted stop DID land left it
    behind, retried at every start for a prompt already gone. The stop's own
    answer now decides: settled clears it, unsettled keeps it."""
    params = {"prompt": "waves", "width": 720, "height": 1280, "seconds": 4, "seed": 1}

    fake = _SettlingFake(stop_settles)
    settled: list = []
    with pytest.raises(video.VideoError, match="timed out"):
        await video.render(fake, VIDEO_WORKFLOW, params, poll_interval=0.01, timeout=0.05,
                           on_settled=settled.append)
    assert fake.interrupted == ["prompt-1"]
    assert settled == (["prompt-1"] if stop_settles else [])

    fake = _SettlingFake(stop_settles)
    settled = []
    task = asyncio.create_task(video.render(fake, VIDEO_WORKFLOW, params, on_settled=settled.append))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert settled == (["prompt-1"] if stop_settles else [])


@pytest.mark.asyncio
async def test_a_render_comfyui_finished_reports_settled():
    class Done(_InterruptibleFake):
        async def poll(self, prompt_id):
            return comfy.COMPLETED, 1.0

        async def fetch(self, prompt_id, output_node_id=None):
            return b"mp4", "video/mp4"

    settled: list = []
    assert await video.render(Done(), VIDEO_WORKFLOW, {"prompt": "waves", "width": 720, "height": 1280,
                                              "seconds": 4, "seed": 1},
                              on_settled=settled.append) == (b"mp4", "video/mp4")
    assert settled == ["prompt-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"queue_running": None}, {"queue_running": "p1"},
                                  {"queue_running": [], "queue_pending": {"p1": 1}}])
async def test_a_malformed_queue_answer_is_never_proof_the_prompt_is_gone(body):
    """Copilot on #336: a 200 /queue whose fields were not lists read as
    "not running, not pending", so interrupt() reported the stop settled and
    startup forgot a prompt that may still be rendering. Absence is only
    evidence from a queue that was actually listed. (A missing queue_pending
    alone reads as empty, as older answers omit it.)"""
    def handler(request):
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(200, json=body)
        return httpx.Response(200, json={})

    assert await _client(handler).interrupt("p1") is False


@pytest.mark.asyncio
async def test_an_older_queue_answer_without_pending_still_reads():
    seen: list = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": []})
        return httpx.Response(200, json={})

    assert await _client(handler).interrupt("p1") is True


@pytest.mark.asyncio
async def test_a_failing_record_during_a_cancelled_submit_still_stops_the_prompt():
    """Copilot on #336: on the cancel-during-submit path, a record callback
    that raised skipped the stop, leaving an accepted prompt running with no
    marker. The stop now runs whatever the record does."""
    fake = _SlowSubmitFake()
    backend = video.VideoBackend(fake, {"t2v": VIDEO_WORKFLOW}, timeout=60.0)

    def failing(prompt_id):
        raise OSError("disk full")

    task = asyncio.create_task(backend.render(
        "t2v", {"prompt": "waves", "width": 720, "height": 1280, "seconds": 4, "seed": 1},
        on_submit=failing))
    await fake.accepted.wait()
    task.cancel()
    await asyncio.sleep(0)
    fake.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.interrupted == ["prompt-1"]
