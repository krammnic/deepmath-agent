import json
import os
import unittest
from unittest.mock import MagicMock, patch

from deepmath.cli import expand_environment
from deepmath.providers import ChatCompletionsProvider, ProviderError


class ProviderTests(unittest.TestCase):
    def test_chat_contract_and_role_model(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"claim":"test"}'}}],
                    "model": "resolved-model", "usage": {"total_tokens": 20}}
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
        with patch("deepmath.providers.build_opener", return_value=opener), patch.dict(os.environ, {"TEST_LLM_KEY": "secret-value"}):
            provider = ChatCompletionsProvider(model="configured-model", role_models={"critic": "critic-model"}, api_key_env="TEST_LLM_KEY")
            result = provider.complete("critic", "JSON please", {"data": "problem"})
        self.assertEqual(result.data, {"claim": "test"})
        request = opener.open.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(sent["model"], "critic-model")
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertNotIn("secret-value", str(sent))

    def test_truncated_and_malformed_outputs_fail(self):
        for finish, content in [("length", '{}'), ("stop", 'not-json'), ("stop", '[]')]:
            opener = MagicMock()
            opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(
                {"choices": [{"finish_reason": finish, "message": {"content": content}}]}).encode()
            with patch("deepmath.providers.build_opener", return_value=opener):
                with self.assertRaises(ProviderError):
                    ChatCompletionsProvider(model="test", api_key_env="").complete("theorist", "JSON", {})

    def test_credentials_required_without_network(self):
        with patch.dict(os.environ, {}, clear=True), patch("deepmath.providers.build_opener") as opener:
            with self.assertRaisesRegex(ProviderError, "Set OPENAI_API_KEY"):
                ChatCompletionsProvider(model="test").complete("theorist", "JSON", {})
            opener.assert_not_called()

    def test_local_endpoint_and_remote_tls(self):
        ChatCompletionsProvider(model="test", base_url="http://localhost:8000/v1", api_key_env="")
        for url in ["http://external.example/v1", "https://key@host/v1", "https://host/v1?key=secret"]:
            with self.assertRaises(ValueError):
                ChatCompletionsProvider(model="test", base_url=url)

    def test_environment_placeholders_fail_explicitly(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPMATH_MODEL"):
                expand_environment({"model": "${DEEPMATH_MODEL}"})


if __name__ == "__main__":
    unittest.main()
