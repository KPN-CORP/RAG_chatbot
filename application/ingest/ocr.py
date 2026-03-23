import boto3
import time
from pathlib import Path
from typing import BinaryIO, Optional, Union, List

import os
from dotenv import load_dotenv

load_dotenv()
session = boto3.Session()

def extract_pdf_pages(
    pdf_path: str,
    *,
    s3_bucket: str,
    s3_key: str,
    region: Optional[str] = None,
    poll_interval: float = 2.0,
    timeout_sec: int = 60 * 30,
) -> str:
    """
    Extract plain text from a scanned PDF using AWS Textract async API.
    Designed for 50+ page PDFs.

    Returns:
        merged_text (str) with [[PAGE=N]] markers
    """

    region = region or os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
    s3_bucket = os.getenv("S3_BUCKET_NAME") 

    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    textract = boto3.client(
        "textract",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # 1) Upload PDF to S3
    s3.upload_file(pdf_path, s3_bucket, s3_key)

    # 2) Start async Textract job
    start_resp = textract.start_document_text_detection(
        DocumentLocation={
            "S3Object": {
                "Bucket": s3_bucket,
                "Name": s3_key,
            }
        }
    )
    job_id = start_resp["JobId"]

    # 3) Poll until job finishes
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

    # 4) Fetch ALL pages (pagination)
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


def extract(image: Union[str, Path, bytes, bytearray, BinaryIO]) -> dict:
    image_binary = image_to_binary(image)

    client = session.client('textract', region_name=os.getenv("AWS_DEFAULT_REGION"), aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    # client = boto3.client('textract')
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
    pdf_path: str, s3_bucket: str, s3_key: str
) -> List[dict]:
    """Extract LINE blocks from Textract for scanned PDFs."""
    resp = extract_pdf_pages(pdf_path=pdf_path, s3_bucket=s3_bucket, s3_key=s3_key)
    return [block for block in resp if isinstance(block, dict) and block.get("BlockType") == "LINE"]