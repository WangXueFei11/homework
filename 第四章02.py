from datetime import datetime
today = datetime.now()
Today = datetime.utcnow()
someday = datetime(2023,10,9,15,12,1)
print(today)
print(Today)
print(someday.isoweekday())
print(today.strftime("%Y-%m-%d %H :%M :%S"))
