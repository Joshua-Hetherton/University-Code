Create database SchoolDB;
Use SchoolDB;

Create table Students(
StudentID INT Primary Key,
FirstName Varchar(30),
LastName Varchar(30),
DateOfBirth DATE,
GradeLevel INT

);

Create table Courses(
CourseID INT Primary Key,
CourseName Varchar(30),
Credits INT,
TeacherID INT,

Foreign Key (TeacherID) References Teachers(TeacherID)
);

Alter database SchoolDB drop Courses;
Create table Enrollments(
EnrollmentID INT Primary Key,
StudentID INT,
CourseID INT,
EnrollmentDate DATE,

foreign key (StudentID) References Students(StudentID),
foreign key (CourseID) References Courses(CourseID)

);

Alter Table Students ADD email varchar(30);
Alter table Students Rename Column GradeLevel To ClassLevel;
Alter Table Students Drop Column email;
Alter Table Courses ADD Constraint CHECK(Credits Between 1 AND 5);

-- drop database schooldb;


Create table Teachers(
TeacherID INT Primary Key,
Firstname Varchar(50) Not Null,
Lastname Varchar(50) Not Null,
Department Varchar(50)

);

DROP Table Enrollments;
DROP Table Courses;




