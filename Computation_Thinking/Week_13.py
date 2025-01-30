
import datetime

def loo_visted(birthdate):
    today = datetime.date.today()
    return print(f"You have approximatley been to the loo {((today - birthdate)*6).days} times")

def meals_eaten(birthdate):
    today = datetime.date.today()
    return print(f"You have approximatley eaten {((today - birthdate)*3).days} meals")

def days_slept(birthdate):
    today = datetime.date.today()
    return print(f"You have slept for {(((today - birthdate))*0.33).days} days")

def print_duration(birthdate):
    today = datetime.date.today()
    difference = today - birthdate
    print(f'You have been alive for {difference.days} days.')
    
    


def get_birthdate():
    year = int(input('What year were you born? '))
    month = int(input('What month were you born? '))
    day = int(input('What day were you born? '))
    return datetime.date(year, month, day)


birthdate = get_birthdate()
print_duration(birthdate)
days_slept(birthdate)
meals_eaten(birthdate)
loo_visted(birthdate)
