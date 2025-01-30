def get_date():
    user_input=input("Enter date of workout ")
    return user_input

def get_duration():
    user_input=input("Enter a duration of the workout ")
    if(user_input.isdigit()):
        return int(user_input)
    else:
        print("Not valid, try again")

def get_workout():
    valid_entry=False
    while(not valid_entry):
        user_input=input("Enter workout type ")
        if (user_input.lower() in valid_workouts):
            valid_entry=True
            return user_input
        else:
            print("Try Again")

def add_workout():
    workouts.append([get_date(),get_workout(),get_duration()])
    print("Workout Added")
    pass

def find_workout():
    found_workouts=[]
    print(workouts)
    user_input=get_workout()
    for index in range(len(workouts)):

        if(workouts[index][1]==user_input):
            found_workouts.append([workouts[index][0],workouts[index][2]])
    return found_workouts()

def bulk_upload():
    filename="Week_13\workouts group 1.csv"
    in_file=open(filename,"r")

    index=0
    for line in in_file:
        split_line=line.split(",")
        workouts.append([split_line[0],split_line[1].lower(),split_line[2]])
        index+=1

    in_file.close()
    print("File Uploaded")

def calc_average():
    specific_workout=find_workout()
    sum=0
    for index in range(len(specific_workout())):
        sum+=specific_workout[index][2]
    return sum/len(specific_workout)

def output_to_file():
    shortest_duration=min(list(workouts[index][2] for index in range(len(workouts))))
    longest_duration=max(list(workouts[index][2] for index in range(len(workouts))))

    shortest_workout=workouts.index(shortest_duration)
    longest_workout=workouts.index(longest_duration)




    filename="Week_13\Output_workout.txt"
    in_file=open(filename,"w")
    in_file.write(f"")
    in_file.close()



valid_workouts=["strength training","running(indoors)","running(outdoors)","swimming(indoors)","swimming(outdoors)","gym","zumba","spin class"]



workouts=[["12/12/12","gym",3],["12/12/12","spin class",3],["12/12/12","spin class",5]]

loop=False
while(not loop):
    user_input=input("Enter what to do: Add workout, Find Workout")
    if(user_input=="1"):

        add_workout()
    elif(user_input=="2"):
        print(find_workout())
    elif(user_input=="3"):
        bulk_upload()
    elif(user_input=="4"):
        print(calc_average())
    elif(user_input=="5"):
        output_to_file()


    elif(user_input=="Exit"):
        loop=True
    

