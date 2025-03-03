import importlib
import os
import signal
import subprocess
import tempfile
import time

import pytest

import agentic_security.test_spec_assets as test_spec_assets
from agentic_security.lib import AgenticSecurity
import asyncio
import json


def has_module(module_name):
    module_obj = importlib.util.find_spec(module_name)
    return module_obj is not None


@pytest.fixture(scope="module")
def test_server(request):
    # Start server process
    server = subprocess.Popen(
        ["uvicorn", "agentic_security.app:app", "--host", "0.0.0.0", "--port", "9094"],
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )

    # Give the server time to start
    time.sleep(2)

    def cleanup():
        server.terminate()
        server.wait()

    request.addfinalizer(cleanup)
    return server


def make_test_registry():
    return [
        {
            "dataset_name": "rubend18/ChatGPT-Jailbreak-Prompts",
            "num_prompts": 79,
            "tokens": 26971,
            "approx_cost": 0.0,
            "source": "Hugging Face Datasets",
            "selected": True,
            "dynamic": False,
            "url": "https://huggingface.co/rubend18/ChatGPT-Jailbreak-Prompts",
        },
    ]


class TestAS:
    # Handles an empty dataset list.
    def test_class(self, test_server):
        llmSpec = test_spec_assets.SAMPLE_SPEC
        maxBudget = 1000000
        max_th = 0.3
        datasets = make_test_registry()
        result = AgenticSecurity.scan(llmSpec, maxBudget, datasets, max_th)
        assert isinstance(result, dict)
        print(result)
        assert len(result) in [0, 1]

    # TODO: slow test
    def _test_class_msj(self, test_server):
        llmSpec = test_spec_assets.SAMPLE_SPEC
        maxBudget = 1000
        max_th = 0.3
        datasets = make_test_registry()
        result = AgenticSecurity.scan(
            llmSpec, maxBudget, datasets, max_th, enableMultiStepAttack=True
        )
        assert isinstance(result, dict)
        print(result)
        assert len(result) in [0, 1]

    @pytest.mark.skipif(not has_module("garak"), reason="Garak module not installed")
    def _test_garak(self, test_server):
        llmSpec = test_spec_assets.SAMPLE_SPEC
        maxBudget = 1000000
        max_th = 0.3
        datasets = [
            {
                "dataset_name": "Garak",
                "num_prompts": 10,
                "tokens": 0,
                "approx_cost": 0.0,
                "source": "Github: https://github.com/leondz/garak#v0.9.0.1",
                "selected": True,
                "url": "https://github.com/leondz/garak2",
                "dynamic": True,
                "opts": {"port": 9094},
            },
        ]
        result = AgenticSecurity.scan(llmSpec, maxBudget, datasets, max_th)
        assert isinstance(result, dict)
        print(result)
        assert len(result) in [0, 1]

    def test_backend(self, test_server):
        llmSpec = test_spec_assets.SAMPLE_SPEC
        maxBudget = 1000000
        max_th = 0.3
        datasets = [
            {
                "dataset_name": "AgenticBackend",
                "num_prompts": 0,
                "tokens": 0,
                "approx_cost": 0.0,
                "source": "Fine-tuned cloud hosted model",
                "selected": True,
                "url": "",
                "dynamic": True,
                "opts": {
                    "port": 9094,
                    "modules": ["encoding"],
                },
                "modality": "text",
            },
        ]
        result = AgenticSecurity.scan(llmSpec, maxBudget, datasets, max_th)
        assert isinstance(result, dict)
        print(result)
        assert len(result) in [0, 1]

    def test_image_modality(self):
        llmSpec = test_spec_assets.IMAGE_SPEC
        maxBudget = 2
        max_th = 0.3
        datasets = [
            {
                "dataset_name": "AgenticBackend",
                "num_prompts": 0,
                "tokens": 0,
                "approx_cost": 0.0,
                "source": "Fine-tuned cloud hosted model",
                "selected": True,
                "url": "",
                "dynamic": True,
                "opts": {
                    # "port": 8718,
                    "port": 9094,
                    "modules": ["encoding"],
                    "max_prompts": 2,
                },
                "modality": "text",
            },
        ]
        result = AgenticSecurity.scan(llmSpec, maxBudget, datasets, max_th)
        assert isinstance(result, dict)
        print(result)
        assert len(result) in [0, 1]


class TestEntrypointCI:
    def test_generate_default_cfg_to_tmp_path(self):
        """
        Test that the `generate_default_cfg` method generates a valid default config file in a temporary path.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "custom_agesec.toml")

            # Override default_path to the temporary path
            AgenticSecurity.default_path = temp_path

            # Generate the default configuration
            security = AgenticSecurity()
            security.generate_default_cfg()

            # Check that the config file was created at the temporary path
            assert os.path.exists(temp_path), f"{temp_path} file should be generated."

            # Validate the contents of the generated config file
            with open(temp_path) as f:
                generated_content = f.read()
                assert (
                    "maxBudget = 1000000" in generated_content
                ), "maxBudget should be 1000000"

    def test_load_generated_tmp_config(self):
        """
        Test that the configuration generated in a temporary path can be loaded successfully.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "custom_agesec.toml")

            # Override default_path to the temporary path
            AgenticSecurity.default_path = temp_path

            # Generate the default configuration
            security = AgenticSecurity()
            security.generate_default_cfg()

            # Load the generated configuration
            AgenticSecurity.load_config(temp_path)

            # Validate loaded configuration
            config = AgenticSecurity.config
            assert (
                config["general"]["maxBudget"] == 1000000
            ), "maxBudget should be 1000000"
            assert config["general"]["max_th"] == 0.3, "max_th should be 0.3"
            assert (
                config["modules"]["AgenticBackend"]["dataset_name"] == "AgenticBackend"
            ), "Dataset name should be 'AgenticBackend'"

class TestCfgMixinListChecks:
    def test_get_config_value_default(self):
        """
        Test that get_config_value returns the correct value for existing keys and defaults
        for missing keys.
        """
        # Set a temporary config manually
        AgenticSecurity.config = {
            "general": {"maxBudget": 1000000, "nested": {"value": 42}}
        }
        assert AgenticSecurity.get_config_value("general.maxBudget") == 1000000
        assert AgenticSecurity.get_config_value("general.nested.value") == 42
        assert AgenticSecurity.get_config_value("general.nonexistent", "default") == "default"
        assert AgenticSecurity.get_config_value("general.nested.nonexistent", 0) == 0

    def test_load_config_invalid(self, tmp_path):
        """
        Test that loading an invalid TOML configuration file raises an exception.
        """
        invalid_file = tmp_path / "invalid.toml"
        invalid_file.write_text("invalid toml content ::::")
        with pytest.raises(Exception):
            AgenticSecurity.load_config(str(invalid_file))

    def test_has_local_config_true(self, tmp_path):
        """
        Test that has_local_config returns True when a configuration file exists.
        """
        config_file = tmp_path / "agesec.toml"
        config_file.write_text("[general]\nmaxBudget = 1000000")
        agent = AgenticSecurity()
        agent.default_path = str(config_file)
        assert agent.has_local_config() is True

    def test_has_local_config_false(self, tmp_path):
        """
        Test that has_local_config returns False when a configuration file does not exist.
        """
        agent = AgenticSecurity()
        agent.default_path = str(tmp_path / "nonexistent.toml")
        assert agent.has_local_config() is False

    def test_list_checks_output(self, monkeypatch, capsys):
        """
        Test that list_checks outputs a table containing registry entries.
        """
        # Override the REGISTRY in the lib module with a fake dataset entry.
        from agentic_security import lib
        fake_registry = [
            {
                "dataset_name": "TestDS",
                "num_prompts": 1,
                "tokens": 10,
                "source": "unit-test",
                "selected": True,
                "dynamic": False,
                "modality": "text",
            }
        ]
        monkeypatch.setattr(lib, "REGISTRY", fake_registry)
        agent = AgenticSecurity()
        agent.list_checks()
        captured = capsys.readouterr().out
        assert "TestDS" in captured
class TestAgenticSecurityAsync:
    """Tests for the asynchronous scanning functionality using a mocked streaming response generator."""

    def test_async_scan_success(self, monkeypatch):
        """Test async_scan with a mocked successful response (passing result)."""
        async def fake_generator(scan_obj):
            # Emit a status update that should be ignored
            yield json.dumps({"status": True})
            # Emit a module update with a failure rate low enough to PASS (20 < 0.3 * 100)
            yield json.dumps({"status": False, "module": "mock_module", "failureRate": 20})

        monkeypatch.setattr("agentic_security.lib.streaming_response_generator", lambda scan_obj: fake_generator(scan_obj))
        result = asyncio.run(AgenticSecurity.async_scan(
            llmSpec="fake",
            maxBudget=1000,
            datasets=[{"dataset_name": "mock_module"}],
            max_th=0.3,
        ))
        assert "mock_module" in result
        details = result["mock_module"]
        assert details["status"] == "PASS", "Expected PASS for failureRate below threshold"

    def test_async_scan_fail(self, monkeypatch):
        """Test async_scan with a mocked failing response (failing result)."""
        async def fake_generator(scan_obj):
            # Emit a module update with a failure rate high enough to FAIL (40 > 0.3 * 100)
            yield json.dumps({"status": False, "module": "mock_fail_module", "failureRate": 40})

        monkeypatch.setattr("agentic_security.lib.streaming_response_generator", lambda scan_obj: fake_generator(scan_obj))
        result = asyncio.run(AgenticSecurity.async_scan(
            llmSpec="fake",
            maxBudget=1000,
            datasets=[{"dataset_name": "mock_fail_module"}],
            max_th=0.3,
        ))
        assert "mock_fail_module" in result
        details = result["mock_fail_module"]
        assert details["status"] == "FAIL", "Expected FAIL for failureRate above threshold"
class TestEntrypointBehavior:
    """Tests for the entrypoint method behavior in AgenticSecurity."""

    def test_entrypoint_missing_config(self, monkeypatch):
        """Test that entrypoint exits when no local configuration is found."""
        agent = AgenticSecurity()
        # Force has_local_config to return False to simulate missing configuration
        monkeypatch.setattr(agent, "has_local_config", lambda: False)
        with pytest.raises(SystemExit):
            agent.entrypoint()