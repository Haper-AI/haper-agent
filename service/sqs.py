import boto3
from util.env import RuntimeEnv

sqs_client = boto3.client(
    'sqs',
    region_name=RuntimeEnv.Instance().SQS_REGION,
    endpoint_url=RuntimeEnv.Instance().SQS_ENDPOINT,
    aws_access_key_id=RuntimeEnv.Instance().AWS_ACCESS_KEY_ID,
    aws_secret_access_key=RuntimeEnv.Instance().AWS_SECRET_ACCESS_KEY,
)