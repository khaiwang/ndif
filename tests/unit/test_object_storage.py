"""Unit tests for ObjectStorageMixin, TelemetryMixin, and TensorStoragePickler.

Tests the S3 object storage mixin used by BackendRequestModel,
BackendResponseModel, and BackendResultModel for persistence.
"""

import logging
import pickle
from io import BytesIO
from typing import ClassVar
from unittest.mock import MagicMock, call, patch

import pytest
from pydantic import ConfigDict

from src.common.schema.mixins import (
    ObjectStorageMixin,
    TelemetryMixin,
    TensorStoragePickler,
    cpu_pickle_module,
)

_MIXINS_MODULE = "src.common.schema.mixins"


# ---------------------------------------------------------------------------
# Test subclasses — concrete implementations of ObjectStorageMixin
# ---------------------------------------------------------------------------


class JsonModel(ObjectStorageMixin):
    """Test model using JSON serialization."""

    _folder_name: ClassVar[str] = "test-json"
    _file_extension: ClassVar[str] = "json"

    id: str
    name: str = ""
    value: int = 0


class PtModel(ObjectStorageMixin):
    """Test model using PyTorch serialization."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    _folder_name: ClassVar[str] = "test-pt"
    _file_extension: ClassVar[str] = "pt"

    id: str


class TelemetryModel(TelemetryMixin):
    """Test model for TelemetryMixin."""

    pass


# ---------------------------------------------------------------------------
# Shared fixture: mock ObjectStoreProvider
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_store():
    """Mock ObjectStoreProvider for all tests in this module."""
    with patch(f"{_MIXINS_MODULE}.ObjectStoreProvider") as mock_provider:
        mock_client = MagicMock()
        mock_provider.object_store = mock_client
        mock_provider.object_store_bucket = "test-bucket"
        yield {
            "provider": mock_provider,
            "client": mock_client,
        }


# =========================================================================
# object_name()
# =========================================================================


class TestObjectName:
    """Tests for ObjectStorageMixin.object_name() key construction."""

    def test_json_format(self):
        """JSON model produces {folder}/{id}.json key."""
        assert JsonModel.object_name("req-001") == "test-json/req-001.json"

    def test_pt_format(self):
        """PyTorch model produces {folder}/{id}.pt key."""
        assert PtModel.object_name("res-042") == "test-pt/res-042.pt"


# =========================================================================
# url()
# =========================================================================


class TestUrl:
    """Tests for ObjectStorageMixin.url() presigned URL generation."""

    def test_generates_presigned_url(self, mock_store):
        """Calls generate_presigned_url with correct params."""
        mock_store["client"].generate_presigned_url.return_value = (
            "https://s3/presigned"
        )
        model = JsonModel(id="req-001", name="test")

        result = model.url()

        mock_store["client"].generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={
                "Bucket": "test-bucket",
                "Key": "test-json/req-001.json",
            },
            ExpiresIn=7200,
        )
        assert result == "https://s3/presigned"


# =========================================================================
# _save()
# =========================================================================


class TestInternalSave:
    """Tests for ObjectStorageMixin._save() internal upload method."""

    def test_uploads_to_s3(self, mock_store):
        """Uploads data with correct bucket, key, and content type."""
        model = JsonModel(id="req-001", name="test")
        data = BytesIO(b'{"test": true}')

        model._save(data, "application/json")

        mock_store["client"].upload_fileobj.assert_called_once()
        call_kwargs = mock_store["client"].upload_fileobj.call_args
        assert call_kwargs[1]["Bucket"] == "test-bucket"
        assert call_kwargs[1]["Key"] == "test-json/req-001.json"
        assert call_kwargs[1]["ExtraArgs"] == {
            "ContentType": "application/json"
        }

    def test_checks_bucket_exists(self, mock_store):
        """Calls head_bucket to verify bucket exists before upload."""
        model = JsonModel(id="req-001")
        model._save(BytesIO(b"data"), "application/json")

        mock_store["client"].head_bucket.assert_called_once_with(
            Bucket="test-bucket"
        )

    def test_creates_bucket_on_client_error(self, mock_store):
        """Creates bucket when head_bucket raises ClientError."""
        # Must use a real exception class for the except clause to catch it
        client_error = type("ClientError", (Exception,), {})
        mock_store["client"].exceptions.ClientError = client_error
        mock_store["client"].head_bucket.side_effect = client_error()
        model = JsonModel(id="req-001")

        model._save(BytesIO(b"data"), "application/json")

        mock_store["client"].create_bucket.assert_called_once_with(
            Bucket="test-bucket"
        )
        # Upload should still proceed
        mock_store["client"].upload_fileobj.assert_called_once()

    def test_seeks_to_start_before_upload(self, mock_store):
        """Resets BytesIO position to 0 before uploading."""
        model = JsonModel(id="req-001")
        data = BytesIO(b"some data")
        data.seek(5)  # Move to middle

        model._save(data, "application/json")

        # The data passed to upload_fileobj should be seeked to 0
        uploaded = mock_store["client"].upload_fileobj.call_args[1]["Fileobj"]
        assert uploaded.tell() == 0


# =========================================================================
# _load()
# =========================================================================


class TestInternalLoad:
    """Tests for ObjectStorageMixin._load() internal download method."""

    def test_loads_bytes(self, mock_store):
        """Downloads and returns object data as bytes."""
        mock_body = MagicMock()
        mock_body.read.return_value = b'{"name": "test"}'
        mock_store["client"].get_object.return_value = {
            "Body": mock_body,
            "ContentLength": 17,
        }

        result = JsonModel._load("req-001")

        mock_store["client"].get_object.assert_called_once_with(
            Bucket="test-bucket", Key="test-json/req-001.json"
        )
        assert result == b'{"name": "test"}'
        mock_body.close.assert_called_once()

    def test_streaming_returns_body_and_length(self, mock_store):
        """Streaming mode returns (StreamingBody, content_length) tuple."""
        mock_body = MagicMock()
        mock_store["client"].get_object.return_value = {
            "Body": mock_body,
            "ContentLength": 1024,
        }

        body, length = JsonModel._load("req-001", stream=True)

        assert body is mock_body
        assert length == 1024
        # Body should NOT be read or closed in streaming mode
        mock_body.read.assert_not_called()
        mock_body.close.assert_not_called()


# =========================================================================
# save() — JSON path
# =========================================================================


class TestSaveJson:
    """Tests for ObjectStorageMixin.save() with JSON serialization."""

    def test_serializes_as_json(self, mock_store):
        """JSON models use model_dump_json() for serialization."""
        model = JsonModel(id="req-001", name="hello", value=42)

        result = model.save()

        # Verify upload was called
        mock_store["client"].upload_fileobj.assert_called_once()
        call_kwargs = mock_store["client"].upload_fileobj.call_args[1]
        assert call_kwargs["ExtraArgs"]["ContentType"] == "application/json"

        # Verify data content
        uploaded_data = call_kwargs["Fileobj"]
        uploaded_data.seek(0)
        content = uploaded_data.read().decode("utf-8")
        assert '"name":"hello"' in content or '"name": "hello"' in content
        assert result is model  # Returns self for chaining

    def test_tracks_size(self, mock_store):
        """_size is set to the byte length of serialized data."""
        model = JsonModel(id="req-001", name="test", value=1)
        model.save()
        assert model._size > 0

    def test_returns_self(self, mock_store):
        """save() returns self for method chaining."""
        model = JsonModel(id="req-001")
        assert model.save() is model


# =========================================================================
# save() — PyTorch path
# =========================================================================


class TestSavePt:
    """Tests for ObjectStorageMixin.save() with PyTorch serialization."""

    @patch(f"{_MIXINS_MODULE}.torch")
    def test_uses_torch_save(self, mock_torch, mock_store):
        """PyTorch models use torch.save with cpu_pickle_module."""
        model = PtModel(id="res-001")

        model.save()

        mock_torch.save.assert_called_once()
        args, kwargs = mock_torch.save.call_args
        assert kwargs["pickle_module"] is cpu_pickle_module

    @patch(f"{_MIXINS_MODULE}.torch")
    def test_payload_excludes_id_and_size(self, mock_torch, mock_store):
        """Payload passed to torch.save excludes 'id' and '_size'."""
        model = PtModel(id="res-001")

        model.save()

        payload = mock_torch.save.call_args[0][0]
        assert "id" not in payload
        assert "_size" not in payload

    @patch(f"{_MIXINS_MODULE}.zstd")
    @patch(f"{_MIXINS_MODULE}.torch")
    def test_compression(self, mock_torch, mock_zstd, mock_store):
        """compress=True applies zstd compression at level 6."""
        # Make torch.save write some data to the buffer
        def fake_save(payload, buf, **kwargs):
            buf.write(b"fake tensor data")

        mock_torch.save.side_effect = fake_save

        mock_compressor = MagicMock()
        mock_writer = MagicMock()
        mock_writer.__enter__ = MagicMock(return_value=mock_writer)
        mock_writer.__exit__ = MagicMock(return_value=False)
        mock_compressor.stream_writer.return_value = mock_writer
        mock_zstd.ZstdCompressor.return_value = mock_compressor

        model = PtModel(id="res-001")
        model.save(compress=True)

        mock_zstd.ZstdCompressor.assert_called_once_with(level=6)

    @patch(f"{_MIXINS_MODULE}.torch")
    def test_content_type_octet_stream(self, mock_torch, mock_store):
        """PyTorch saves use application/octet-stream content type."""
        model = PtModel(id="res-001")
        model.save()

        call_kwargs = mock_store["client"].upload_fileobj.call_args[1]
        assert call_kwargs["ExtraArgs"]["ContentType"] == "application/octet-stream"


# =========================================================================
# load()
# =========================================================================


class TestLoad:
    """Tests for ObjectStorageMixin.load() deserialization."""

    def test_json_deserializes_to_instance(self, mock_store):
        """JSON load returns a model instance."""
        json_data = b'{"id": "req-001", "name": "loaded", "value": 99}'
        mock_body = MagicMock()
        mock_body.read.return_value = json_data
        mock_store["client"].get_object.return_value = {
            "Body": mock_body,
            "ContentLength": len(json_data),
        }

        result = JsonModel.load("req-001")

        assert isinstance(result, JsonModel)
        assert result.id == "req-001"
        assert result.name == "loaded"
        assert result.value == 99

    @patch(f"{_MIXINS_MODULE}.torch")
    def test_pt_deserializes_to_dict(self, mock_torch, mock_store):
        """PyTorch load returns a dict via torch.load."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"fake pt data"
        mock_store["client"].get_object.return_value = {
            "Body": mock_body,
            "ContentLength": 12,
        }
        mock_torch.load.return_value = {"key": "value"}

        result = PtModel.load("res-001")

        assert result == {"key": "value"}
        mock_torch.load.assert_called_once()
        _, kwargs = mock_torch.load.call_args
        assert kwargs["map_location"] == "cpu"
        assert kwargs["weights_only"] is False

    def test_streaming_passthrough(self, mock_store):
        """stream=True returns raw (body, length) without deserializing."""
        mock_body = MagicMock()
        mock_store["client"].get_object.return_value = {
            "Body": mock_body,
            "ContentLength": 2048,
        }

        result = JsonModel.load("req-001", stream=True)

        assert result == (mock_body, 2048)


# =========================================================================
# delete()
# =========================================================================


class TestDelete:
    """Tests for ObjectStorageMixin.delete() S3 deletion."""

    def test_deletes_object(self, mock_store):
        """Calls delete_object with correct bucket and key."""
        JsonModel.delete("req-001")

        mock_store["client"].delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="test-json/req-001.json"
        )

    def test_silent_on_error(self, mock_store):
        """Swallows exceptions silently."""
        mock_store["client"].delete_object.side_effect = Exception("not found")

        JsonModel.delete("req-001")  # Should not raise


# =========================================================================
# TensorStoragePickler
# =========================================================================


class TestTensorStoragePickler:
    """Tests for TensorStoragePickler GPU→CPU tensor handling."""

    @patch(f"{_MIXINS_MODULE}.torch")
    def test_gpu_tensor_moved_to_cpu(self, mock_torch):
        """GPU tensors are detached and moved to CPU before pickling."""
        mock_tensor = MagicMock()
        mock_torch.is_tensor.return_value = True
        mock_tensor.device.type = "cuda"
        mock_cpu_tensor = MagicMock()
        mock_tensor.detach.return_value.to.return_value = mock_cpu_tensor
        mock_cpu_tensor.__reduce_ex__ = MagicMock(
            return_value=(None, (), None)
        )

        pickler = TensorStoragePickler(BytesIO())
        result = pickler.reducer_override(mock_tensor)

        mock_tensor.detach.assert_called_once()
        mock_tensor.detach.return_value.to.assert_called_once_with("cpu")
        mock_cpu_tensor.__reduce_ex__.assert_called_once_with(
            pickle.HIGHEST_PROTOCOL
        )
        assert result is not NotImplemented

    @patch(f"{_MIXINS_MODULE}.torch")
    def test_cpu_tensor_passthrough(self, mock_torch):
        """CPU tensors return NotImplemented (default pickle)."""
        mock_tensor = MagicMock()
        mock_torch.is_tensor.return_value = True
        mock_tensor.device.type = "cpu"

        pickler = TensorStoragePickler(BytesIO())
        result = pickler.reducer_override(mock_tensor)

        assert result is NotImplemented

    @patch(f"{_MIXINS_MODULE}.torch")
    def test_non_tensor_passthrough(self, mock_torch):
        """Non-tensor objects return NotImplemented (default pickle)."""
        mock_torch.is_tensor.return_value = False

        pickler = TensorStoragePickler(BytesIO())
        result = pickler.reducer_override("just a string")

        assert result is NotImplemented


# =========================================================================
# cpu_pickle_module
# =========================================================================


class TestCpuPickleModule:
    """Tests for the cpu_pickle_module wrapper."""

    def test_pickler_is_tensor_storage_pickler(self):
        """cpu_pickle_module.Pickler is TensorStoragePickler."""
        assert cpu_pickle_module.Pickler is TensorStoragePickler

    def test_has_standard_pickle_attrs(self):
        """Module exposes standard pickle attributes."""
        assert hasattr(cpu_pickle_module, "dumps")
        assert hasattr(cpu_pickle_module, "loads")
        assert hasattr(cpu_pickle_module, "HIGHEST_PROTOCOL")


# =========================================================================
# TelemetryMixin
# =========================================================================


class TestTelemetryMixin:
    """Tests for TelemetryMixin.backend_log() logging utility."""

    def test_info_level(self):
        """Logs at info level."""
        model = TelemetryModel()
        logger = MagicMock(spec=logging.Logger)

        result = model.backend_log(logger, "test message", level="info")

        logger.info.assert_called_once_with("test message")
        assert result is model

    def test_error_level(self):
        """Logs at error level."""
        model = TelemetryModel()
        logger = MagicMock(spec=logging.Logger)

        model.backend_log(logger, "error msg", level="error")

        logger.error.assert_called_once_with("error msg")

    def test_exception_level(self):
        """Logs at exception level."""
        model = TelemetryModel()
        logger = MagicMock(spec=logging.Logger)

        model.backend_log(logger, "exc msg", level="exception")

        logger.exception.assert_called_once_with("exc msg")

    def test_unknown_level_no_op(self):
        """Unknown level does nothing, returns self."""
        model = TelemetryModel()
        logger = MagicMock(spec=logging.Logger)

        result = model.backend_log(logger, "msg", level="debug")

        logger.info.assert_not_called()
        logger.error.assert_not_called()
        logger.exception.assert_not_called()
        assert result is model
