from service.sqs import sqs_client
from util.env import RuntimeEnv
from util.logger import logger


def handle_message(sqs_message):
    pass


if __name__ == '__main__':
    logger.info('Agent service starting up...')
    while True:
        response = sqs_client.receive_message(
            QueueUrl=RuntimeEnv.Instance().SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10,  # long pooling
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
