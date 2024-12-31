from typing import Optional
from dynamoose import model, Model, Schema

class Transaction(Model):
    class Meta:
        table_name = "transactions"
        timestamps = True

    userId: str
    transactionId: str
    dateTime: str
    courseId: str
    paymentProvider: str  # "stripe"
    amount: Optional[float]
