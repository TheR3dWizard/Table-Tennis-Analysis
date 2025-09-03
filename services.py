import pika
from kafka import KafkaProducer, KafkaConsumer

class RabbitMQService:
    def __init__(self, username, host=None, port=None, queue=None):
        self.host = host or 'localhost'
        self.port = port or 5672
        self.queue = queue or 'default'
        self.connection = None
        self.channel = None
        self.username = username
        self.password = "nopass"  # Password can be set if needed

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue)

    def close(self):
        if self.connection:
            self.connection.close()

    def publish(self, message, queue=None):
        if not self.channel:
            raise Exception("Channel not initialized. Call connect() first.")
        self.channel.basic_publish(exchange='', routing_key=queue or self.queue, body=message)

    def consume(self, callback, queue=None):
        if not self.channel:
            raise Exception("Channel not initialized. Call connect() first.")
        for method_frame, properties, body in self.channel.consume(queue=queue or self.queue):
            callback(body)
            self.channel.basic_ack(method_frame.delivery_tag)
    
    def getlastmessage(self, queue=None):
        if not self.channel:
            raise Exception("Channel not initialized. Call connect() first.")
        method_frame, properties, body = self.channel.basic_get(queue=queue or self.queue)
        if method_frame:
            return body
        return None

class ServiceManager:
    def __init__(self, service_type, config):
        self.service_type = service_type
        self.config = config
        self.service = None

    def start_service(self):
        if self.service_type == 'kafka_producer':
            self.service = KafkaProducer(**self.config)
        elif self.service_type == 'kafka_consumer':
            self.service = KafkaConsumer(**self.config)
        elif self.service_type == 'rabbitmq':
            connection = pika.BlockingConnection(pika.ConnectionParameters(**self.config))
            self.service = connection.channel()
        else:
            raise ValueError("Unsupported service type")

    def stop_service(self):
        if self.service_type == 'kafka_producer':
            self.service.close()
        elif self.service_type == 'kafka_consumer':
            self.service.close()
        elif self.service_type == 'rabbitmq':
            self.service.connection.close()

    def send_message(self, topic, message):
        if self.service_type == 'kafka_producer':
            self.service.send(topic, value=message)
        elif self.service_type == 'rabbitmq':
            self.service.basic_publish(exchange='', routing_key=topic, body=message)
        else:
            raise ValueError("Send message not supported for this service type")
    
    def consume_messages(self, topic):
        if self.service_type == 'kafka_consumer':
            self.service.subscribe([topic])
            for message in self.service:
                yield message.value
        elif self.service_type == 'rabbitmq':
            for method_frame, properties, body in self.service.consume(queue=topic):
                yield body
                self.service.basic_ack(method_frame.delivery_tag)
        else:
            raise ValueError("Consume messages not supported for this service type")
    

    def get_service(self):
        return self.service