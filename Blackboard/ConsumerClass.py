from services import RabbitMQService, PostgresService
import uuid
import requests
import time
import pprint

class Consumer:
    def __init__(
        self,
        name,
        id=None,
        queuename=None,
        rabbitmqusername="default",
        rabbitmqpassword="default",
        serverurl="http://localhost:6060",
    ):
        self.name = name
        self.id = id or str(uuid.uuid4())
        self.queuename = queuename or self.name.lower().replace(" ", "-")
        print(
            f"Created a {self.name} consumer with ID: {self.id} and Queue: {self.queuename}"
        )
        self.rabbitmqservice = RabbitMQService(
            username=rabbitmqusername, password=rabbitmqpassword, queue=self.queuename
        )
        self.rabbitmqservice.connect()
        print(
            f"{self.name} consumer sucessfully connected to RabbitMQ with {rabbitmqusername} credential"
        )
        self.server = serverurl
        self.hashmap = {}
        self.pgs = PostgresService(username="pw1tt", password="securepostgrespassword")
        self.pgs.connect()
        # self.rabbitmqservice.consume(self.messagecallback, self.queuename)

    def threadstart(self):
        self.rabbitmqservice.consume(self.messagecallback, self.queuename)

    def placerequest(self, columnslist, targetid, returnmessageid):
        message = {
            "type": "request",
            "requestid": str(uuid.uuid4()),
            "requesterid": self.id,
            "returnqueue": self.queuename,
            "targetid": targetid,
            "columnslist": columnslist,
            "returnmessageid": returnmessageid
        }
        print(f"Placing request... for {columnslist} to {targetid}")
        # [ABSTRACTED] self.rabbitmqservice.publish(str(message), queue=targetqueue)
        response = requests.post(f"{self.server}/placerequest", json=message)

        return response.json()

    def placesuccess(self, requestid, requesterid, targetid, requestorqueue, returnmessageid):
        print(f"Placing success message... from {self.queuename} to {requestorqueue}")
        message = {
            "type": "success",
            "requestid": requestid,
            "requesterid": requesterid,
            "targetid": targetid,
            "returnqueue": requestorqueue,
            "returnmessageid": returnmessageid
        }
        self.rabbitmqservice.publish(str(message), queue=requestorqueue)

    def messagecallback(self, body):
        body = eval(body)  # convert string to dict

        # Perform callback logic with context specific model
        # print(body)
        if body["type"] == "request":
            print(f"\n\n[REQUEST] Message Received:")
            pprint.pprint(body)
            print("\n\n")

            logicreturn = self.logic(body)
            time.sleep(5)  # simulate processing time
            if logicreturn:
                print("Logic executed successfully, sending success message...")
                self.placesuccess(
                    body["requestid"],
                    body["requesterid"],
                    body["targetid"],
                    body["returnqueue"],
                    body["returnmessageid"]
                )
            else:
                print("Logic execution failed or pending, not sending success message.")

        elif body["type"] == "success":
            print(f"\n\n[SUCCESS] Message Received:")
            pprint.pprint(body)
            print("\n\n")
            if body["returnmessageid"] in self.hashmap:
                messagebody = self.hashmap.pop(body["returnmessageid"])
                self.logic(messagebody)
            else:
                print("Request ID not found in hashmap. Ignoring success message.")

    def check(self, frameid, columnlist):
        message = {"frameid": frameid, "columns": columnlist}

        response = requests.post(f"{self.server}/checkandreturn", json=message)
        return response.json()

    def update(self, frameid, columnname, value):
        message = {"frameid": frameid, "column": columnname, "value": value}

        response = requests.post(f"{self.server}/updatecolumn", json=message)
        return response.json()
