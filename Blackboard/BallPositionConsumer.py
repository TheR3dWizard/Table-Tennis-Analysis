from ConsumerClass import Consumer
import random

class BallPositionConsumer(Consumer):
    def __init__(
        self,
        name="Ball Position Detection",
        id=None,
        queuename=None,
        rabbitmqusername="default",
        rabbitmqpassword="default",
        serverurl="http://localhost:6060",
    ):
        super().__init__(
            name,
            id,
            queuename,
            rabbitmqusername,
            rabbitmqpassword,
            serverurl,
        )
        self.logic = self.logicfunction
        self.testframeid = 16
    
    def computerandomballcoordinates(self):
        print("Executing computerandomballcoordinates....")
        x, y, z = random.uniform(0,1000), random.uniform(0,1000), random.uniform(0,1000)

        return {
            "ballx": x,
            "bally": y,
            "ballz": z,
        }

    def logicfunction(self, messagebody):
        boardresponse = self.check(self.testframeid, ["tablex1", "tabley1", "tablex2", "tabley2", "tablex3", "tabley3", "tablex4", "tabley4"])
        if not boardresponse:
            print("Table vertices not found for the given frame. Cannot compute ball coordinates.")
            self.placerequest(["tablex1", "tabley1", "tablex2", "tabley2", "tablex3", "tabley3", "tablex4", "tabley4"], "Table Vertex Detection", messagebody["requestid"])
            self.hashmap[messagebody["requestid"]] = messagebody
            return False

        computeresult = self.computerandomballcoordinates()
        print(f"Computed ball coordinates: {computeresult}")
        for column, value in computeresult.items():
            self.update(self.testframeid, column, value) # should later use messagebody.frameid
        
        return True
    
if __name__ == "__main__":
    c1 = BallPositionConsumer(rabbitmqusername="pw1tt", rabbitmqpassword="securerabbitmqpassword", id="ball-position-detection-consumer")
    c1.threadstart()