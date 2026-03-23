import os
import time
import uuid
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
session = boto3.Session()



def _aws_client(service_name: str, *, region: str):
    """Create a boto3 client with the credentials configured in the environment."""
    return boto3.client(
        service_name,
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )



def _normalize_bucket_region(location_constraint: Optional[str]) -> str:
    """Normalize the S3 location constraint into a boto3 region name."""
    if not location_constraint:
        return "us-east-1"
    if location_constraint == "EU":
        return "eu-west-1"
    return location_constraint



def _get_bucket_region(s3_client, bucket: str) -> str:
    """Return the real region for an S3 bucket."""
    response = s3_client.get_bucket_location(Bucket=bucket)
    return _normalize_bucket_region(response.get("LocationConstraint"))



def _wait_for_s3_object(s3_client, *, bucket: str, key: str, timeout_sec: int = 30) -> None:
    """Wait until the newly uploaded object is readable via ``head_object``."""
    start_time = time.time()
    while True:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return
        except ClientError as exc:
            error_code = (exc.response.get("Error") or {}).get("Code")
            if error_code not in {"404", "NoSuchKey", "NotFound"}:
                raise
            if time.time() - start_time > timeout_sec:
                raise TimeoutError(
                    f"Timed out waiting for s3://{bucket}/{key} to become readable"
                ) from exc
            time.sleep(1.0)



def extract_pdf_pages(
    pdf_path: str,
    *,
    s3_bucket: str,
    s3_key: Optional[str] = None,
    region: Optional[str] = None,
    poll_interval: float = 2.0,
    timeout_sec: int = 60 * 30,
    cleanup_s3_object: bool = True,
) -> List[dict]:
    """Extract Textract blocks from a scanned PDF using a temporary S3 object.

    Textract async PDF APIs require the document to live in S3. This helper
    uploads the PDF to the configured bucket, waits until the object metadata is
    visible, runs Textract, fetches all paginated blocks, and optionally deletes
    the temporary S3 object afterward.
    """

    s3_bucket = s3_bucket or os.getenv("S3_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_BUCKET_NAME must be configured to run Textract OCR fallback")

    region = region or os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
    s3_key = s3_key or f"textract-input/{Path(pdf_path).stem}-{uuid.uuid4().hex}.pdf"

    s3 = _aws_client("s3", region=region)
    bucket_region = _get_bucket_region(s3, s3_bucket)
    if bucket_region != region:
        region = bucket_region
        s3 = _aws_client("s3", region=region)
    textract = _aws_client("textract", region=region)

    try:
        s3.upload_file(pdf_path, s3_bucket, s3_key)
        _wait_for_s3_object(s3, bucket=s3_bucket, key=s3_key)

        start_resp = textract.start_document_text_detection(
            DocumentLocation={
                "S3Object": {
                    "Bucket": s3_bucket,
                    "Name": s3_key,
                }
            }
        )
        job_id = start_resp["JobId"]

        start_time = time.time()
        while True:
            resp = textract.get_document_text_detection(JobId=job_id, MaxResults=1)
            status = resp.get("JobStatus")

            if status == "SUCCEEDED":
                break
            if status == "FAILED":
                raise RuntimeError(f"Textract failed: {resp.get('StatusMessage', '')}")

            if time.time() - start_time > timeout_sec:
                raise TimeoutError(f"Textract timed out (job_id={job_id})")

            time.sleep(poll_interval)

        blocks: List[dict] = []
        next_token: Optional[str] = None

        while True:
            params = {"JobId": job_id, "MaxResults": 1000}
            if next_token:
                params["NextToken"] = next_token

            page = textract.get_document_text_detection(**params)
            blocks.extend(page.get("Blocks", []))
            next_token = page.get("NextToken")

            if not next_token:
                break

        return blocks
    except ClientError as exc:
        error_code = (exc.response.get("Error") or {}).get("Code")
        if error_code == "InvalidS3ObjectException":
            raise RuntimeError(
                "Textract could not read the temporary S3 object. Verify that the bucket "
                f"exists in region '{region}', the credentials can read/write it, and "
                "Textract is allowed to access the uploaded object metadata."
            ) from exc
        raise
    finally:
        if cleanup_s3_object:
            try:
                s3.delete_object(Bucket=s3_bucket, Key=s3_key)
            except Exception:
                pass



def extract(image: Union[str, Path, bytes, bytearray, BinaryIO]) -> dict:
    image_binary = image_to_binary(image)

    client = session.client('textract', region_name=os.getenv("AWS_DEFAULT_REGION"), aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    response = client.detect_document_text(Document={'Bytes': image_binary})
    return response



def image_to_binary(image: Union[str, Path, bytes, bytearray, BinaryIO]) -> bytes:
    if isinstance(image, (str, Path)):
        with open(image, "rb") as file_pointer:
            return file_pointer.read()
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if hasattr(image, "read"):
        return image.read()
    raise TypeError(
        "Unsupported image type. Provide a file path, bytes, bytearray, or a binary file object."
    )



def _extract_textract_lines(
    pdf_path: str, s3_bucket: str, s3_key: Optional[str] = None
) -> List[dict]:
    """Return Textract ``LINE`` blocks for a scanned PDF.

    Args:
        pdf_path: Local PDF path uploaded to S3 for Textract processing.
        s3_bucket: Bucket used as temporary storage for the PDF.
        s3_key: Optional object key used for the uploaded PDF. When omitted, a
            unique temporary key is generated automatically.

    Returns:
        A list of Textract blocks filtered down to ``BlockType == "LINE"``.
        Each block still includes the original page number and geometry so the
        caller can rebuild page text in reading order.
    """
    resp = extract_pdf_pages(pdf_path=pdf_path, s3_bucket=s3_bucket, s3_key=s3_key)
    return [block for block in resp if isinstance(block, dict) and block.get("BlockType") == "LINE"]
