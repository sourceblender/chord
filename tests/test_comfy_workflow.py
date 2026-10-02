"""The public image contract accepts an operator-owned ComfyUI graph."""

import asyncio
import json

import httpx
import pytest

from chord import comfy
from chord.comfy_workflow import ComfyImageBackend, ComfyImageWorkflow, WorkflowError


def workflow_file(tmp_path, graph):
    path = tmp_path / "image-api.json"
    path.write_text(json.dumps(graph))
    return path


def graph():
    return {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "original", "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 7, "model": ["4", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }


def test_fills_only_declared_inputs_without_mutating_operator_workflow(tmp_path):
    original = graph()
    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, original),
        prompt_node_id="6", output_node_id="9", seed_node_id="3",
    )

    first = workflow.for_request('a "glazed" mug', seed=42)
    second = workflow.for_request("a lake", seed=43)

    assert first["6"]["inputs"]["text"] == 'a "glazed" mug'
    assert first["3"]["inputs"]["seed"] == 42
    assert first["6"]["inputs"]["clip"] == ["4", 1]
    assert second["6"]["inputs"]["text"] == "a lake"
    assert workflow.graph == original
    assert workflow.output_node_id == "9"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda g: g.pop("9"), "output node"),
        (lambda g: g["6"]["inputs"].pop("text"), "no input"),
        (lambda g: g["3"]["inputs"].pop("seed"), "no input"),
        (lambda g: g["6"].pop("class_type"), "API-format"),
    ],
)
def test_invalid_binding_or_graph_refuses_before_submission(tmp_path, change, message):
    value = graph()
    change(value)
    with pytest.raises(WorkflowError, match=message):
        ComfyImageWorkflow.load(
            workflow_file(tmp_path, value),
            prompt_node_id="6", output_node_id="9", seed_node_id="3",
        )


def test_workflow_without_seed_binding_keeps_its_fixed_seed(tmp_path):
    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()), prompt_node_id="6", output_node_id="9"
    )
    assert workflow.for_request("a mug")["3"]["inputs"]["seed"] == 7
    with pytest.raises(WorkflowError, match="no seed binding"):
        workflow.for_request("a mug", seed=42)


@pytest.mark.parametrize("seed", [True, 1.0, -1, 2**64])
def test_seed_must_be_a_real_nonnegative_integer(tmp_path, seed):
    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()),
        prompt_node_id="6", output_node_id="9", seed_node_id="3",
    )
    with pytest.raises(WorkflowError, match="image seed"):
        workflow.for_request("a mug", seed=seed)


def test_non_api_format_workflow_refuses(tmp_path):
    with pytest.raises(WorkflowError, match="API-format"):
        ComfyImageWorkflow.load(
            workflow_file(tmp_path, {"nodes": [{"id": 6}]}),
            prompt_node_id="6", output_node_id="9",
        )


@pytest.mark.asyncio
async def test_fetch_reads_only_configured_output_node():
    def handler(request):
        if request.url.path == "/history/p1":
            return httpx.Response(200, json={"p1": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {
                    "9": {"images": [{"filename": "wanted.png"}]},
                    "10": {"images": [{"filename": "unrelated.png"}]},
                },
            }})
        if request.url.path == "/view":
            return httpx.Response(200, content=request.url.params["filename"].encode())
        raise AssertionError(request.url)

    http = httpx.AsyncClient(
        base_url="http://comfy.invalid", transport=httpx.MockTransport(handler)
    )
    client = comfy.ComfyClient("http://comfy.invalid", client=http)
    assert await client.fetch_pngs("p1", "9") == [b"wanted.png"]
    with pytest.raises(comfy.ComfyError, match="output node"):
        await client.fetch_pngs("p1", "11")
    await client.aclose()


@pytest.mark.asyncio
async def test_backend_submits_filled_graph_and_returns_one_bound_png(tmp_path):
    class Fake:
        def __init__(self):
            self.graph = None
            self.output = None

        async def submit(self, graph):
            self.graph = graph
            return "prompt-42"

        async def poll(self, prompt_id):
            assert prompt_id == "prompt-42"
            return comfy.COMPLETED, 1.0

        async def fetch_pngs(self, prompt_id, output_node_id=None):
            self.output = output_node_id
            return [b"png"]

        async def interrupt(self, prompt_id):
            raise AssertionError("completed work must not be interrupted")

    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()),
        prompt_node_id="6", output_node_id="9", seed_node_id="3",
    )
    fake = Fake()
    data, prompt_id = await ComfyImageBackend(workflow, fake).render("a mug", seed=17)
    assert (data, prompt_id) == (b"png", "prompt-42")
    assert fake.graph["6"]["inputs"]["text"] == "a mug"
    assert fake.graph["3"]["inputs"]["seed"] == 17
    assert fake.output == "9"


@pytest.mark.asyncio
async def test_backend_refuses_multiple_outputs_without_resubmitting(tmp_path):
    class Fake:
        submissions = 0

        async def submit(self, graph):
            self.submissions += 1
            return "p1"

        async def poll(self, prompt_id):
            return comfy.COMPLETED, 1.0

        async def fetch_pngs(self, prompt_id, output_node_id=None):
            return [b"one", b"two"]

        async def interrupt(self, prompt_id):
            raise AssertionError("finished work must not be interrupted")

    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()), prompt_node_id="6", output_node_id="9"
    )
    fake = Fake()
    with pytest.raises(WorkflowError, match="exactly one"):
        await ComfyImageBackend(workflow, fake).render("a mug")
    assert fake.submissions == 1


@pytest.mark.asyncio
async def test_failed_comfy_job_cannot_succeed_from_a_partial_png(tmp_path):
    class Fake:
        async def submit(self, graph):
            return "p1"

        async def poll(self, prompt_id):
            return comfy.FAILED, 1.0

        async def fetch_pngs(self, prompt_id, output_node_id=None):
            raise AssertionError("failed history must not be served")

        async def interrupt(self, prompt_id):
            raise AssertionError("failed job has already stopped")

    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()), prompt_node_id="6", output_node_id="9"
    )
    with pytest.raises(WorkflowError, match="failed on ComfyUI"):
        await ComfyImageBackend(workflow, Fake()).render("a mug")


@pytest.mark.asyncio
async def test_repeated_cancel_during_poll_keeps_accepted_prompt_stop_alive(tmp_path):
    class Fake:
        def __init__(self):
            self.polling = asyncio.Event()
            self.stop_started = asyncio.Event()
            self.stop_finished = asyncio.Event()
            self.release_stop = asyncio.Event()
            self.stopped = []

        async def submit(self, graph):
            return "p1"

        async def poll(self, prompt_id):
            self.polling.set()
            await asyncio.Event().wait()
            raise AssertionError("poll must be cancelled")

        async def fetch_pngs(self, prompt_id, output_node_id=None):
            raise AssertionError("cancelled job must not fetch")

        async def interrupt(self, prompt_id):
            self.stop_started.set()
            await self.release_stop.wait()
            self.stopped.append(prompt_id)
            self.stop_finished.set()
            return True

    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()), prompt_node_id="6", output_node_id="9"
    )
    fake = Fake()
    backend = ComfyImageBackend(workflow, fake)
    task = asyncio.create_task(backend.render("a mug"))
    await fake.polling.wait()
    task.cancel()
    await fake.stop_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fake.release_stop.set()
    await asyncio.wait_for(fake.stop_finished.wait(), timeout=1)
    assert fake.stopped == ["p1"]


@pytest.mark.asyncio
async def test_image_waits_for_shared_render_lock_before_submitting(tmp_path):
    class Fake:
        def __init__(self):
            self.submitted = asyncio.Event()

        async def submit(self, graph):
            self.submitted.set()
            return "p1"

        async def poll(self, prompt_id):
            return comfy.COMPLETED, 1.0

        async def fetch_pngs(self, prompt_id, output_node_id=None):
            return [b"png"]

        async def interrupt(self, prompt_id):
            raise AssertionError("completed job must not be stopped")

    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()), prompt_node_id="6", output_node_id="9"
    )
    fake = Fake()
    lock = asyncio.Lock()
    await lock.acquire()
    task = asyncio.create_task(ComfyImageBackend(workflow, fake, render_lock=lock).render("a mug"))
    await asyncio.sleep(0)
    assert not fake.submitted.is_set()
    lock.release()
    assert await task == (b"png", "p1")
    assert fake.submitted.is_set()


@pytest.mark.asyncio
async def test_cancel_during_submit_stops_the_accepted_prompt(tmp_path):
    class Fake:
        def __init__(self):
            self.submitting = asyncio.Event()
            self.accept = asyncio.Event()
            self.stopped_event = asyncio.Event()
            self.stopped = []

        async def submit(self, graph):
            self.submitting.set()
            await self.accept.wait()
            return "accepted-prompt"

        async def poll(self, prompt_id):
            raise AssertionError("cancelled submission must not poll")

        async def fetch_pngs(self, prompt_id, output_node_id=None):
            raise AssertionError("cancelled submission must not fetch")

        async def interrupt(self, prompt_id):
            self.stopped.append(prompt_id)
            self.stopped_event.set()
            return True

    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()), prompt_node_id="6", output_node_id="9"
    )
    fake = Fake()
    task = asyncio.create_task(ComfyImageBackend(workflow, fake).render("a mug"))
    await fake.submitting.wait()
    task.cancel()
    fake.accept.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.stopped == ["accepted-prompt"]


@pytest.mark.asyncio
async def test_second_cancel_cannot_orphan_an_accepted_prompt(tmp_path):
    class Fake:
        def __init__(self):
            self.submitting = asyncio.Event()
            self.accept = asyncio.Event()
            self.stopped = asyncio.Event()
            self.interrupted = []

        async def submit(self, graph):
            self.submitting.set()
            await self.accept.wait()
            return "accepted-prompt"

        async def poll(self, prompt_id):
            raise AssertionError("cancelled submission must not poll")

        async def fetch_pngs(self, prompt_id, output_node_id=None):
            raise AssertionError("cancelled submission must not fetch")

        async def interrupt(self, prompt_id):
            self.interrupted.append(prompt_id)
            self.stopped.set()
            return True

    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()), prompt_node_id="6", output_node_id="9"
    )
    fake = Fake()
    backend = ComfyImageBackend(workflow, fake)
    task = asyncio.create_task(backend.render("a mug"))
    await fake.submitting.wait()
    task.cancel()
    await asyncio.sleep(0)  # let the first cancellation start independent cleanup
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(backend._cleanup_tasks) == 1
    fake.accept.set()
    await asyncio.wait_for(fake.stopped.wait(), timeout=1)
    assert fake.interrupted == ["accepted-prompt"]
    await asyncio.sleep(0)
    assert not backend._cleanup_tasks


@pytest.mark.asyncio
async def test_timeout_stops_by_prompt_id_and_never_retries(tmp_path):
    class Fake:
        def __init__(self):
            self.submissions = 0
            self.stopped = []

        async def submit(self, graph):
            self.submissions += 1
            return "slow-prompt"

        async def poll(self, prompt_id):
            return comfy.RUNNING, 0.5

        async def fetch_pngs(self, prompt_id, output_node_id=None):
            raise AssertionError("unfinished work must not fetch")

        async def interrupt(self, prompt_id):
            self.stopped.append(prompt_id)
            return True

    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()), prompt_node_id="6", output_node_id="9"
    )
    fake = Fake()
    with pytest.raises(TimeoutError, match="image workflow timed out"):
        await ComfyImageBackend(workflow, fake, timeout_s=0).render("a mug")
    assert fake.submissions == 1
    assert fake.stopped == ["slow-prompt"]


@pytest.mark.asyncio
async def test_timeout_waiting_for_shared_gpu_lock_never_submits(tmp_path):
    class Fake:
        submissions = 0

        async def submit(self, graph):
            self.submissions += 1
            return "unexpected"

    workflow = ComfyImageWorkflow.load(
        workflow_file(tmp_path, graph()), prompt_node_id="6", output_node_id="9"
    )
    lock = asyncio.Lock()
    await lock.acquire()
    fake = Fake()
    backend = ComfyImageBackend(workflow, fake, timeout_s=0.02, render_lock=lock)
    try:
        with pytest.raises(TimeoutError, match="image workflow timed out"):
            await asyncio.wait_for(backend.render("a mug"), timeout=0.5)
        assert fake.submissions == 0
        assert lock.locked()
    finally:
        lock.release()
