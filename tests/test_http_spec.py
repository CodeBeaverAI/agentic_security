import asyncio
import base64
import httpx
import pytest

from agentic_security.http_spec import (
    LLMSpec,
    parse_http_spec,
    encode_image_base64_by_url,
    encode_audio_base64_by_url,
    escape_special_chars_for_json,
    InvalidHTTPSpecError,
)

class DummyResponse:
    """A dummy HTTP response for testing."""
    def __init__(self, status_code=200, content=b"dummy", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def text(self):
        return self.content.decode("utf-8")

class DummyAsyncClient:
    """A dummy async client to simulate HTTP requests."""
    async def request(self, method, url, headers=None, content=None, files=None, timeout=None):
        return DummyResponse(status_code=200, content=b"ok")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class DummyAsyncClientContext:
    """A dummy async client context manager to replace httpx.AsyncClient."""
    def __init__(self, *args, **kwargs):
        self.client = DummyAsyncClient()

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc, tb):
        pass

@pytest.fixture(autouse=True)
def patch_async_client(monkeypatch):
    """Patch httpx.AsyncClient to use DummyAsyncClientContext for every async call."""
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyAsyncClientContext())

@pytest.fixture(autouse=True)
def patch_httpx_get(monkeypatch):
    """Patch httpx.get for encode_image_base64_by_url and encode_audio_base64_by_url."""
    def dummy_get(url, *args, **kwargs):
        # Return dummy content based on url type.
        if "audio" in url:
            return type("DummyResponse", (), {"content": b"audio"})()
        else:
            return type("DummyResponse", (), {"content": b"image"})()
    monkeypatch.setattr(httpx, "get", dummy_get)

def test_parse_http_spec_image_audio_and_files():
    """Test parsing an HTTP spec with placeholders for image, audio and files."""
    http_spec = (
        "POST http://example.com\n"
        "Content-Type: multipart/form-data\n"
        "\n"
        "This is a body with <<BASE64_IMAGE>> and <<BASE64_AUDIO>>."
    )
    spec = parse_http_spec(http_spec)
    assert spec.method == "POST"
    assert spec.url == "http://example.com"
    assert spec.headers == {"Content-Type": "multipart/form-data"}
    assert spec.body == "This is a body with <<BASE64_IMAGE>> and <<BASE64_AUDIO>>."
    assert spec.has_files is True
    assert spec.has_image is True
    assert spec.has_audio is True

def test_escape_special_chars_for_json():
    """Test the escaping of special characters in a prompt."""
    prompt = 'This is a "test" prompt with special chars: \\ \n \r \t'
    escaped = escape_special_chars_for_json(prompt)
    expected = 'This is a \\"test\\" prompt with special chars: \\\\ \\n \\r \\t'
    assert escaped == expected

def test_validate_errors():
    """Test that the validate method raises ValueError when required parameters are missing."""
    spec = LLMSpec(
        method="GET",
        url="http://example.com",
        headers={},
        body="<<BASE64_IMAGE>> <<BASE64_AUDIO>>",
        has_files=True,
        has_image=True,
        has_audio=True,
    )
    # Test missing files.
    with pytest.raises(ValueError, match="Files are required for this request."):
        spec.validate("prompt", encoded_image="image", encoded_audio="audio", files={})

    # Test missing image.
    spec.has_files = False
    with pytest.raises(ValueError, match="An image is required for this request."):
        spec.validate("prompt", encoded_image="", encoded_audio="audio", files={})

    # Test missing audio.
    spec.has_image = False
    spec.has_audio = True
    with pytest.raises(ValueError, match="Audio is required for this request."):
        spec.validate("prompt", encoded_image="image", encoded_audio="", files={})

@pytest.mark.asyncio
async def test_probe_text_mode():
    """Test the probe method in text mode (without image, audio, or files)."""
    spec = LLMSpec(
        method="POST",
        url="http://example.com",
        headers={"Content-Type": "application/json"},
        body='{"prompt": "<<PROMPT>>"}',
    )
    response = await spec.probe("hello")
    assert response.status_code == 200
    # Ensure that the prompt placeholder was used (the dummy client always returns "ok").

@pytest.mark.asyncio
async def test_probe_with_files():
    """Test the probe method when files are provided."""
    spec = LLMSpec(
        method="POST",
        url="http://example.com",
        headers={"Content-Type": "multipart/form-data"},
        body="test",
        has_files=True,
    )
    # Provide dummy files dictionary.
    response = await spec.probe("prompt", files={"file": ("dummy.txt", b"dummy")})
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_verify_with_image():
    """Test the verify method when an image is required in the spec."""
    spec = LLMSpec(
        method="POST",
        url="http://example.com",
        headers={"Content-Type": "text/plain"},
        body="contains <<BASE64_IMAGE>>",
        has_image=True,
    )
    response = await spec.verify()
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_verify_with_audio():
    """Test the verify method when audio is required in the spec."""
    spec = LLMSpec(
        method="POST",
        url="http://example.com",
        headers={"Content-Type": "text/plain"},
        body="contains <<BASE64_AUDIO>>",
        has_audio=True,
    )
    response = await spec.verify()
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_verify_with_files():
    """Test the verify method when files are required."""
    spec = LLMSpec(
        method="POST",
        url="http://example.com",
        headers={"Content-Type": "multipart/form-data"},
        body="test",
        has_files=True,
    )
    response = await spec.verify()
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_from_string_invalid():
    """Test that LLMSpec.from_string can parse a minimal valid spec with only a method and URL."""
    minimal_spec = "INVALID SPEC"
    spec = LLMSpec.from_string(minimal_spec)
    assert spec.method == "INVALID"
    assert spec.url == "SPEC"
    assert spec.headers == {}
    assert spec.body == ""
    assert spec.has_files is False
    assert spec.has_image is False
    assert spec.has_audio is False
@pytest.mark.asyncio
async def test_encode_image_base64_by_url():
    """Test that encode_image_base64_by_url returns properly encoded image data."""
    result = encode_image_base64_by_url("http://dummy-image-url.com")
    expected_prefix = "data:image/jpeg;base64,"
    expected_data = base64.b64encode(b"image").decode("utf-8")
    assert result == expected_prefix + expected_data

@pytest.mark.asyncio
async def test_encode_audio_base64_by_url():
    """Test that encode_audio_base64_by_url returns properly encoded audio data."""
    result = encode_audio_base64_by_url("http://dummy-audio-url.com/audio.mp3")
    expected_prefix = "data:audio/mpeg;base64,"
    expected_data = base64.b64encode(b"audio").decode("utf-8")
    assert result == expected_prefix + expected_data

def test_from_string_empty():
    """Test that LLMSpec.from_string raises an error with an empty specification."""
    with pytest.raises(Exception) as excinfo:
        LLMSpec.from_string("")
    assert "Failed to parse HTTP spec" in str(excinfo.value)

def test_modality_property():
    """Test the modality property for text, image, and audio flags."""
    # When no modality is required, default to TEXT.
    spec_text = LLMSpec(method="GET", url="http://example.com", headers={}, body="Test")
    assert spec_text.modality.name == "TEXT"

    # When an image is required.
    spec_image = LLMSpec(method="GET", url="http://example.com", headers={}, body="<<BASE64_IMAGE>>", has_image=True)
    assert spec_image.modality.name == "IMAGE"

    # When only audio is required.
    spec_audio = LLMSpec(method="GET", url="http://example.com", headers={}, body="<<BASE64_AUDIO>>", has_audio=True)
    assert spec_audio.modality.name == "AUDIO"

    # When both image and audio flags are true, image takes precedence.
    spec_both = LLMSpec(method="GET", url="http://example.com", headers={}, body="<<BASE64_IMAGE>> <<BASE64_AUDIO>>", has_image=True, has_audio=True)
    assert spec_both.modality.name == "IMAGE"

def test_escape_special_chars_for_empty_string():
    """Test that escaping an empty string returns an empty string."""
@pytest.mark.asyncio
async def test_escape_only_backslashes():
    """Test that escape_special_chars_for_json correctly escapes a string of only backslashes."""
    input_str = r"\\"
    # The function first escapes every "\" into "\\".
    # For an input of two backslashes, the result should have four.
    expected = r"\\\\"
    result = escape_special_chars_for_json(input_str)
    assert result == expected
    assert escape_special_chars_for_json("") == ""

@pytest.mark.asyncio
async def test_probe_placeholder_replacement(monkeypatch):
    """Test that the probe method correctly replaces all placeholders in the request body."""
    # Create a capturing async client to intercept the request parameters.
    captured = {}

    class CaptureAsyncClient:
        async def request(self, method, url, headers=None, content=None, files=None, timeout=None):
            captured['method'] = method
            captured['url'] = url
            captured['headers'] = headers
            captured['content'] = content
            captured['files'] = files
            captured['timeout'] = timeout
            return DummyResponse(status_code=200, content=b"ok")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Override httpx.AsyncClient with our capture client.
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: CaptureAsyncClient())

    body_template = "Prompt: <<PROMPT>>, Image: <<BASE64_IMAGE>>, Audio: <<BASE64_AUDIO>>"
    spec = LLMSpec(method="POST", url="http://example.com", headers={"Content-Type": "application/json"}, body=body_template)
    prompt = 'Hello "Test"\nNewLine'
    encoded_image = "img_data"
    encoded_audio = "audio_data"
    await spec.probe(prompt, encoded_image=encoded_image, encoded_audio=encoded_audio)
    # Verify that the placeholders have been replaced correctly.
    escaped_prompt = escape_special_chars_for_json(prompt)
    expected_content = body_template.replace("<<PROMPT>>", escaped_prompt).replace("<<BASE64_IMAGE>>", encoded_image).replace("<<BASE64_AUDIO>>", encoded_audio)
    assert captured.get('content') == expected_content
    assert captured.get('method') == spec.method
@pytest.mark.asyncio
async def test_parse_http_spec_multiline_body():
    """Test parsing an HTTP spec with a multi-line body.
    This verifies that parse_http_spec concatenates the body lines without inserting newlines.
    """
    http_spec = (
        "PUT http://example.com/resource\n"
        "Content-Type: application/json\n"
        "\n"
        "{\n"
        "    \"message\": \"Hello\",\n"
        "    \"status\": \"ok\"\n"
        "}"
    )
    spec = parse_http_spec(http_spec)
    # Since the parser concatenates lines after the header, the expected body will be:
    expected_body = "{    \"message\": \"Hello\",    \"status\": \"ok\"}"
    assert spec.method == "PUT"
    assert spec.url == "http://example.com/resource"
    assert spec.headers == {"Content-Type": "application/json"}
    assert spec.body == expected_body

def test_parse_http_spec_header_with_colon():
    """Test parsing an HTTP spec with headers that include a colon in the header value."""
    http_spec = "GET http://example.com\nAuthorization: Bearer token:extra\n\nBody"
    spec = parse_http_spec(http_spec)
    assert spec.method == "GET"
    assert spec.url == "http://example.com"
    assert spec.headers == {"Authorization": "Bearer token:extra"}
    assert spec.body == "Body"

@pytest.mark.asyncio
async def test_fn_alias(monkeypatch):
    """Test that the 'fn' attribute is an alias of the probe method and works correctly."""
    spec = LLMSpec(
        method="POST",
        url="http://example.com",
        headers={"Content-Type": "application/json"},
        body='{"prompt": "<<PROMPT>>"}'
    )
    captured = {}

    class CaptureClient:
        async def request(self, method, url, headers=None, content=None, files=None, timeout=None):
            captured['content'] = content
            return DummyResponse(status_code=200, content=b"ok")
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: CaptureClient())
    prompt = "alias test"
    response = await spec.fn(prompt)
    assert response.status_code == 200
    expected_prompt = escape_special_chars_for_json(prompt)
    expected_content = spec.body.replace("<<PROMPT>>", expected_prompt)
    assert captured.get("content") == expected_content

@pytest.mark.asyncio
async def test_probe_without_placeholder(monkeypatch):
    """Test that the probe method leaves the body intact when there are no placeholders.
    In this case even when a prompt is provided, since there is no <<PROMPT>> in the body,
    the probe function should send the static content unchanged.
    """
    body = "Static content with no placeholders."
    spec = LLMSpec(
        method="POST",
        url="http://example.com",
        headers={"Content-Type": "text/plain"},
        body=body
    )
    captured = {}

    class CaptureStaticClient:
        async def request(self, method, url, headers=None, content=None, files=None, timeout=None):
            captured['content'] = content
            return DummyResponse(status_code=200, content=b"ok")
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: CaptureStaticClient())
    response = await spec.probe("ignored_prompt")
    assert captured.get("content") == body
@pytest.mark.asyncio
async def test_probe_exception_propagation(monkeypatch):
    """Test that the probe method propagates exceptions raised during the HTTP request."""
    class FailingAsyncClient:
        async def request(self, method, url, headers=None, content=None, files=None, timeout=None):
            raise httpx.RequestError("Simulated request failure")

        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: FailingAsyncClient())
    spec = LLMSpec(
        method="POST",
        url="http://example.com",
        headers={"Content-Type": "application/json"},
        body='{"prompt": "<<PROMPT>>"}'
    )
    with pytest.raises(httpx.RequestError, match="Simulated request failure"):
        await spec.probe("trigger exception")