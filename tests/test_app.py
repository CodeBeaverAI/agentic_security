import asyncio
import pytest
from fastapi import FastAPI
from agentic_security.core.app import create_app, get_tools_inbox, get_stop_event, get_current_run, set_current_run, tools_inbox, stop_event, current_run

# Test create_app returns a FastAPI instance
def test_create_app():
    """Test if create_app returns a FastAPI instance."""
    app = create_app()
    assert isinstance(app, FastAPI)

# Test get_tools_inbox returns the global queue instance with expected behavior
def test_get_tools_inbox():
    """Test the tools_inbox global Queue: it should initially be empty and support enqueueing and dequeueing."""
    queue = get_tools_inbox()
    # Initially the queue should be empty
    assert queue.empty()
    # Put an item and check that the queue is no longer empty
    queue.put_nowait("test_item")
    assert not queue.empty()
    # Remove the item and validate
    item = queue.get_nowait()
    assert item == "test_item"

# Test get_stop_event returns the global stop event and that it can be set
def test_get_stop_event():
    """Test that the stop event returned is initially not set and can be set correctly."""
    event = get_stop_event()
    # Initially the event should not be set
    assert not event.is_set()
    # Set the event and verify it's set
    event.set()
    assert event.is_set()

# Test get_current_run returns default global run dictionary
def test_get_current_run_default():
    """Test get_current_run returns the default state of the current_run global dictionary."""
    # Reset the current_run for consistency
    current_run["spec"] = ""
    current_run["id"] = ""
    run = get_current_run()
    assert isinstance(run, dict)
    assert run.get("spec") == ""
    assert run.get("id") == ""

# Test set_current_run updates the global state correctly
def test_set_current_run():
    """Test that set_current_run correctly updates the current_run global dictionary."""
    spec_value = "test_spec"
    updated_run = set_current_run(spec_value)
    # Ensure that the spec is updated to the given value
    assert updated_run["spec"] == spec_value
    # Ensure that the id is computed as hash(id(spec_value))
    expected_id = hash(id(spec_value))
    assert updated_run["id"] == expected_id

# Test that global state persists across function calls
def test_global_state_persistence():
    """Test that updating global state persists across successive calls."""
    spec_value = "persistent_spec"
    set_current_run(spec_value)
    run1 = get_current_run()
    assert run1["spec"] == spec_value
    spec_value2 = "new_spec"
    set_current_run(spec_value2)
    run2 = get_current_run()
    assert run2["spec"] == spec_value2

# Cleanup fixture: reset global state after each test to avoid state interference
@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global objects (current_run and tools_inbox) after each test."""
    yield
    current_run["spec"] = ""
    current_run["id"] = ""
    while not tools_inbox.empty():
        tools_inbox.get_nowait()
    # Note: asyncio.Event cannot be reset after being set, so we leave stop_event as is.
@pytest.mark.asyncio
async def test_tools_inbox_async():
    """Test async put and get on the global tools_inbox queue."""
    queue = get_tools_inbox()
    # Put an item asynchronously
    await queue.put("async_test_item")
    # Get the item asynchronously and validate its content
    item = await queue.get()
    assert item == "async_test_item"

def test_stop_event_clear():
    """Test that the stop_event can be cleared using its .clear() method."""
    event = get_stop_event()
    # Set the event and verify it is set
    event.set()
    assert event.is_set()
    # Clear the event and verify that it is no longer set
    event.clear()
    assert not event.is_set()

def test_global_current_run_object():
    """Test that the global current_run dictionary remains the same object across function calls."""
    run_initial = get_current_run()
    spec_value = "object_test_spec"
    # Call set_current_run and then get_current_run again; the underlying object should be identical
    set_current_run(spec_value)
    run_updated = get_current_run()
    assert run_initial is run_updated
    # Validate that the spec was updated accordingly
    assert run_updated["spec"] == spec_value
def test_tools_inbox_singleton():
    """Test that get_tools_inbox returns the same global queue instance as the imported variable tools_inbox."""
    assert get_tools_inbox() is tools_inbox

def test_stop_event_singleton():
    """Test that get_stop_event returns the same global event instance as the imported variable stop_event."""
    assert get_stop_event() is stop_event

def test_get_current_run_mutable():
    """Test that the object returned from get_current_run is mutable by updating its value."""
    run = get_current_run()
    run["spec"] = "mutable_test_value"
    # Calling get_current_run again should reflect the mutation
    run_updated = get_current_run()
    assert run_updated["spec"] == "mutable_test_value"

def test_set_current_run_with_none():
    """Test that setting current run with None works as expected."""
    updated_run = set_current_run(None)
    assert updated_run["spec"] is None
    # id(None) returns a constant value in a given run, so we verify it's computed correctly
    assert updated_run["id"] == hash(id(None))

def test_create_app_routes():
    """Test that the FastAPI app returned by create_app has default API docs URLs."""
    app = create_app()
    # FastAPI sets openapi_url and docs_url by default, verify these properties exist and have expected values.
    assert app.openapi_url == "/openapi.json"
    assert app.docs_url == "/docs"
# New tests to increase coverage

def test_set_current_run_with_int():
    """Test set_current_run works with an integer spec value."""
    spec_value = 12345
    updated_run = set_current_run(spec_value)
    assert updated_run["spec"] == spec_value
    assert updated_run["id"] == hash(id(spec_value))

def test_set_current_run_with_list():
    """Test set_current_run works with a list spec value."""
    spec_value = ["item1", "item2"]
    updated_run = set_current_run(spec_value)
    assert updated_run["spec"] == spec_value
    assert updated_run["id"] == hash(id(spec_value))

@pytest.mark.asyncio
async def test_tools_inbox_async_multiple_items():
    """Test async behavior of tools_inbox by concurrently adding and retrieving multiple items."""
    queue = get_tools_inbox()
    items_to_put = [f"item_{i}" for i in range(5)]
    # Concurrently put items into the queue
    await asyncio.gather(*(queue.put(item) for item in items_to_put))
    # Now retrieve items sequentially and verify the order
    retrieved_items = [await queue.get() for _ in items_to_put]
    assert retrieved_items == items_to_put

def test_stop_event_toggle():
    """Test toggling the stop_event by setting and clearing it repeatedly."""
    event = get_stop_event()
    # Make sure to clear any previous state (if supported)
    event.clear()
    assert not event.is_set()
    # Set the event and verify it's set
    event.set()
    assert event.is_set()
    # Clear the event and verify it's no longer set
    event.clear()
    assert not event.is_set()
    # Set it once more to ensure the toggle operation works repeatedly
    event.set()
    assert event.is_set()