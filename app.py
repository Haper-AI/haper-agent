import boto3
from util.env import RuntimeEnv
from util.logger import logger


sqs_client = boto3.client(
    'sqs',
    region_name=RuntimeEnv.Instance().SQS_REGION,
    endpoint_url=RuntimeEnv.Instance().SQS_ENDPOINT,
    aws_access_key_id=RuntimeEnv.Instance().SQS_ACCESS_KEY_ID,
    aws_secret_access_key=RuntimeEnv.Instance().SQS_SECRET_ACCESS_KEY,
)


def handle_message(sqs_message):
    pass


if __name__ == '__main__':
    logger.info('Agent service starting up...')
    while True:
        response = sqs_client.receive_message(
            QueueUrl=RuntimeEnv.Instance().SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10, # long pooling
        )

        messages = response.get('Messages', [])
        if not messages:
            logger.debug("No messages received, continuing...")
            continue


        for message in messages:
            logger.info(f"Received message: {message['MessageId']}")

            # TODO: finish handle message detail and determine different situation for ack messages
            handle_message(message)

            # ACK message
            sqs_client.delete_message(
                QueueUrl=RuntimeEnv.Instance().SQS_QUEUE_URL,
                ReceiptHandle=message['ReceiptHandle']
            )
            logger.info(f"Successfully ACK message: {message['MessageId']}")

