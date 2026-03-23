import sys
import types
from pathlib import Path

if "boto3" not in sys.modules:
    boto3_stub = types.ModuleType("boto3")

    class _DummySession:
        def client(self, *_args, **_kwargs):
            return types.SimpleNamespace()

    boto3_stub.Session = lambda *_args, **_kwargs: _DummySession()
    boto3_stub.client = lambda *_args, **_kwargs: types.SimpleNamespace()
    sys.modules["boto3"] = boto3_stub

if "botocore.exceptions" not in sys.modules:
    botocore_exceptions = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        def __init__(self, response, operation_name):
            super().__init__(operation_name)
            self.response = response
            self.operation_name = operation_name

    botocore_exceptions.ClientError = ClientError
    sys.modules["botocore.exceptions"] = botocore_exceptions

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
    sys.modules["dotenv"] = dotenv_stub

from application.ingest import ocr
from botocore.exceptions import ClientError


class _FakeS3Client:
    def __init__(self, region_name):
        self.region_name = region_name
        self.uploads = []
        self.deleted = []
        self.head_checks = []

    def get_bucket_location(self, Bucket):
        return {"LocationConstraint": "ap-southeast-1"}

    def upload_file(self, filename, bucket, key):
        self.uploads.append((filename, bucket, key))

    def head_object(self, Bucket, Key):
        self.head_checks.append((Bucket, Key))
        return {"ContentLength": 123}

    def delete_object(self, Bucket, Key):
        self.deleted.append((Bucket, Key))


class _FakeTextractClient:
    def __init__(self, region_name):
        self.region_name = region_name
        self.started = []
        self.job_calls = 0

    def start_document_text_detection(self, DocumentLocation):
        self.started.append(DocumentLocation)
        return {"JobId": "job-123"}

    def get_document_text_detection(self, **kwargs):
        self.job_calls += 1
        if kwargs.get("MaxResults") == 1:
            return {"JobStatus": "SUCCEEDED"}
        return {
            "Blocks": [
                {"BlockType": "LINE", "Text": "Scanned text", "Page": 1},
                {"BlockType": "WORD", "Text": "ignored", "Page": 1},
            ]
        }


def test_extract_pdf_pages_uses_temp_object_and_cleans_it_up(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    clients = {}

    def fake_client(service_name, region_name=None, **_kwargs):
        if service_name == "s3":
            client = _FakeS3Client(region_name)
        elif service_name == "textract":
            client = _FakeTextractClient(region_name)
        else:
            raise AssertionError(f"Unexpected service {service_name}")
        clients.setdefault(service_name, []).append(client)
        return client

    monkeypatch.setattr(ocr.boto3, "client", fake_client)

    blocks = ocr.extract_pdf_pages(str(pdf_path), s3_bucket="temp-bucket")

    assert blocks[0]["Text"] == "Scanned text"
    s3_client = clients["s3"][0]
    textract_client = clients["textract"][0]
    upload_bucket, upload_key = s3_client.uploads[0][1], s3_client.uploads[0][2]
    assert upload_bucket == "temp-bucket"
    assert upload_key.startswith("textract-input/sample-")
    assert s3_client.deleted == [("temp-bucket", upload_key)]
    assert textract_client.started[0]["S3Object"]["Name"] == upload_key


def test_extract_pdf_pages_requires_bucket(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)

    try:
        ocr.extract_pdf_pages(str(pdf_path), s3_bucket="")
    except ValueError as exc:
        assert "S3_BUCKET_NAME" in str(exc)
    else:
        raise AssertionError("Expected ValueError when no S3 bucket is configured")


def test_extract_pdf_pages_rewraps_invalid_s3_object(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _BrokenTextract(_FakeTextractClient):
        def start_document_text_detection(self, DocumentLocation):
            raise ClientError({"Error": {"Code": "InvalidS3ObjectException"}}, "StartDocumentTextDetection")

    def fake_client(service_name, region_name=None, **_kwargs):
        if service_name == "s3":
            return _FakeS3Client(region_name)
        if service_name == "textract":
            return _BrokenTextract(region_name)
        raise AssertionError(service_name)

    monkeypatch.setattr(ocr.boto3, "client", fake_client)

    try:
        ocr.extract_pdf_pages(str(pdf_path), s3_bucket="temp-bucket")
    except RuntimeError as exc:
        assert "Textract could not read the temporary S3 object" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for InvalidS3ObjectException")
