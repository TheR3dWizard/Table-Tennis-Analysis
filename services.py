import pika
from kafka import KafkaProducer, KafkaConsumer

class RabbitMQService:
    def __init__(self, host=None, port=None, queue=None):
        self.host = host or 'localhost'
        self.port = port or 5672
        self.queue = queue or 'default'
        self.connection = None
        self.channel = None

    def connect(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host, port=self.port))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue)

    def close(self):
        if self.connection:
            self.connection.close()

    def publish(self, message):
        if not self.channel:
            raise Exception("Channel not initialized. Call connect() first.")
        self.channel.basic_publish(exchange='', routing_key=self.queue, body=message)

    def consume(self, callback):
        if not self.channel:
            raise Exception("Channel not initialized. Call connect() first.")
        for method_frame, properties, body in self.channel.consume(queue=self.queue):
            callback(body)
            self.channel.basic_ack(method_frame.delivery_tag)

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