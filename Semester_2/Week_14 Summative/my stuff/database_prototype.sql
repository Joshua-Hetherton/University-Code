CREATE DATABASE UNIVERSITY_DB;
USE UNIVERSITY_DB;

-- Courses Table
Create Table Courses(
	course_id int primary key auto_increment,
    course_name varchar(100) not nuLl
);


-- Students Table
Create Table Students(
	student_id int primary key auto_increment,
    first_name varchar(50) not null,
    last_name varchar(50) not null,
    email varchar(100) unique not null,
    dob date,
    course_id Int,
    foreign key(course_id) references Courses(course_id)
);


-- Creating Users
create user "admin"@"localhost" identified by "admin@123";
grant all privileges on UNIVERSITY_DB to "admin"@"localhost";

create user "student"@"localhost" identified by "student@123";
grant select on UNIVERSITY_DB.Students to "student"@"localhost";
grant select on UNIVERSITY_DB.Courses to "student"@"localhost";

-- Adding Students and Courses
-- Courses
insert into Courses(course_name) Values
("Computer Science"),
("Software Engineering"),
("Cyber Security"),
("Finance"),
("Archaeology");

-- Students
insert into Students(first_name,last_name, email, dob, course_id) VALUES
("Apollonius", "au Valii-Rath", "apollonius@email.com", "2005-01-01", 1),
("Bob", "Smith", "bob@email.com", "2005-01-02", 2),
("Charlie", "Brown", "charlie@email.com", "2005-01-03", 3),
("Dalinar", "Reed", "dalinar@email.com", "2005-01-04", 4),
("Ephraim", "Cole", "ephraim@email.com", "2005-01-05", 5),
("Fred", "Warren", "fred@email.com", "2005-01-06", 1),
("Gerald", "Morgan", "gerald@email.com", "2005-01-07", 2),
("Henry", "Quinn", "henry@email.com", "2005-01-08", 3),
("Ingrid", "Griffin", "ingrid@email.com", "2005-01-09", 4),
("Julius", "Caesar", "julius@email.com", "2005-01-10", 5),
("Kaladin", "Reynolds", "kaladin@email.com", "2005-01-11", 1),
("Lysander", "Holt", "lysander@email.com", "2005-01-12", 2),
("Matteo", "Nelson", "matteo@email.com", "2005-01-13", 3),
("Nero", "Germanicus", "nero@email.com", "2005-01-14", 4),
("Peter", "Turner", "peter@email.com", "2005-01-15", 5),
("Quinn", "West", "quinn@email.com", "2005-01-16", 1),
("Roque", "Fleming", "roque@email.com", "2005-01-17", 2),
("Sadeas", "Irwin", "sadeas@email.com", "2005-01-18", 3),
("Theodora", "Blake", "theodora@email.com", "2005-01-19", 4),
("Ulysses", "Laertiades", "ulysses@email.com", "2005-01-20", 5);

-- Changing a students information
-- This changes Ulysses email to Odysseus@gmail.com
Update Students
Set email = "Odysseus@gmail.com"
Where student_id=20;

-- Showing a single student
select Students.student_id, Students.first_name, Students.last_name, Students.email, Students.dob, Courses.course_name
from Students, Courses
Where Students.student_id= 17
and Students.course_id = Courses.course_id;

-- Displaying all Students
select Students.student_id, Students.first_name, Students.last_name, Students.email, Students.dob, Courses.course_name
From Students
Join Courses on Students.course_id = Courses.course_id
-- Displays by course and in order of student id
Order by Courses.course_name, Students.student_id;

-- Displaying students of a specific course
select Students.student_id, Students.first_name, Students.last_name, Students.email, Students.dob, Courses.course_name
From Students
Join Courses on Students.course_id = Courses.course_id
Where Students.course_id=5;




