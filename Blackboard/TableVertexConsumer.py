from ConsumerClass import Consumer
import random

class TableVertexConsumer(Consumer):
    def __init__(
        self,
        name="Table Vertex Detection",
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
    
    def computerandomtablevertices(self):
        print("Executing computerandomtablevertices....")
        x1, y1 = random.uniform(0,1000), random.uniform(0,1000)
        x2, y2 = random.uniform(0,1000), random.uniform(0,1000)
        x3, y3 = random.uniform(0,1000), random.uniform(0,1000)
        x4, y4 = random.uniform(0,1000), random.uniform(0,1000)
        '''
        tablex1 FLOAT,
        tabley1 FLOAT,
        tablex2 FLOAT,
        tabley2 FLOAT,
        tablex3 FLOAT,
        tabley3 FLOAT,
        tablex4 FLOAT,
        tabley4 FLOAT,'''
        return {
            "tablex1": x1,
            "tabley1": y1,
            "tablex2": x2,
            "tabley2": y2,
            "tablex3": x3,
            "tabley3": y3,
            "tablex4": x4,
            "tabley4": y4,
        }

    def logicfunction(self, messagebody):
        computeresult = self.computerandomtablevertices()
        print(f"Computed table vertices: {computeresult}")
        c = 0
        for column, value in computeresult.items():
            
            self.update(self.testframeid, column, value) # should later use messagebody.frameid
            c += 1
            print(f"Updated {c} columns in the database for frameid {self.testframeid}")
            
        return True

if __name__ == "__main__":
    c1 = TableVertexConsumer(rabbitmqusername="pw1tt", rabbitmqpassword="securerabbitmqpassword")
    c1.threadstart()