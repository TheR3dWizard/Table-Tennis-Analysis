import psycopg2
import pika


class RabbitMQService:
    def __init__(self, username, password, host=None, port=None, queue=None):
        self.host = host or "localhost"
        self.port = port or 5672
        self.queue = queue or "default"
        self.connection = None
        self.channel = None
        self.username = username
        self.password = password

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host, port=self.port, credentials=credentials
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue)

    def close(self):
        if self.connection:
            self.connection.close()

    def publish(self, message, queue=None):
        if not self.channel:
            raise Exception("Channel not initialized. Call connect() first.")
        self.channel.basic_publish(exchange="", routing_key=queue, body=message)

    def consume(self, callback, queue=None):
        if not self.channel:
            raise Exception("Channel not initialized. Call connect() first.")
        self.channel.basic_consume(
            queue=queue or self.queue,
            on_message_callback=lambda ch, method, properties, body: (
                callback(body),
                ch.basic_ack(delivery_tag=method.delivery_tag),
            ),
        )
        print(
            f" [*] Waiting for messages in '{queue or self.queue}'. To exit press CTRL+C"
        )
        self.channel.start_consuming()

    def getlastmessage(self, queue=None):
        if not self.channel:
            raise Exception("Channel not initialized. Call connect() first.")
        method_frame, properties, body = self.channel.basic_get(
            queue=queue or self.queue
        )
        if method_frame:
            return body
        return None

    def testsanity(self):
        try:
            self.connect()
            self.close()
            return True
        except Exception as e:
            print(f"Sanity test failed: {e}")
            return False


class PostgresService:
    def __init__(self, username, password, host=None, port=None):
        self.host = host or "localhost"
        self.port = port or 5432
        self.database = "blackboard"
        self.username = username
        self.password = password
        self.table = "table_tennis_analysis"
        self.connection = None
        self.VIDEO_TABLE_NAME = "video_data"

    def connect(self):
        self.connection = psycopg2.connect(
            dbname=self.database,
            user=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
        )

    def get_video_path_by_videoid(self, videoid_value):
        if not self.connection:
            self.connect()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT videoPath
                FROM {self.VIDEO_TABLE_NAME}
                WHERE videoId = %s
            """,
                (videoid_value,),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_video_data_by_videoid(self, videoid_value):
        if not self.connection:
            self.connect()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM {self.VIDEO_TABLE_NAME}
                WHERE videoId = %s
            """,
                (videoid_value,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            colnames = [desc[0] for desc in cursor.description]
            return dict(zip(colnames, row))
    
    def add_video_data(self, videoid_value, videopath_value, videoname_value="", videotag_value=""):
        if not self.connection:
            self.connect()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.VIDEO_TABLE_NAME} (videoId, videoPath, videoName, videoTag)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (videoId) DO NOTHING
            """,
                (videoid_value, videopath_value, videoname_value, videotag_value),
            )
            self.connection.commit()

    def get_columns_and_values_by_frameid(self, frameid_value):
        if not self.connection:
            self.connect()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM {self.table}
                WHERE frameid = %s
            """,
                (frameid_value,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            colnames = [desc[0] for desc in cursor.description]
            return dict(zip(colnames, row))

    def has_column_value_by_frameid(self, column_name, frameid_value):
        if not self.connection:
            self.connect()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {column_name}
                FROM {self.table}
                WHERE frameid = %s
            """,
                (frameid_value,),
            )
            result = cursor.fetchone()
            return result is not None and result[0] is not None

    def set_column_value_by_frameid(self, column_name, value, frameid_value):
        if not self.connection:
            self.connect()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self.table} SET {column_name} = %s WHERE frameid = %s",
                (value, frameid_value),
            )
            self.connection.commit()

    def close(self):
        if self.connection:
            self.connection.close()


if __name__ == "__main__":
    # rmqs = RabbitMQService(username='pw1tt', password='securepassword', queue='testqueue')
    # rmqs.connect()
    # rmqs.publish('Hello RabbitMQ!', queue='test2queue')
    # print(rmqs.getlastmessage(queue='testqueue'))

    db = PostgresService("pw1tt", "securepostgrespassword")
    print(db.get_columns_and_values_by_frameid(16))
