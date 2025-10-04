Create database Uni_db;
use Uni_Db;

create table courses (
    course_id int primary key,
    course_name varchar(100)
);

-- Create the students table
create table students (
    student_id int primary key,
    student_name varchar(100)
);

-- Create the grades table
create table grades (
    grade_id int primary key,
    student_id int,
    course_id int,
    grade varchar(2),
    foreign key (student_id) references students(student_id),
    foreign key (course_id) references courses(course_id)
);


Create user 'Professor'@'localhost' identified by 'Prof@123';
Create user 'Student'@'localhost' identified by 'Stud@123';
Create user 'Registrar'@'localhost' identified by 'Reg@123';

-- Professor
grant select on Uni_db.students to 'Professor'@'localhost';
grant select on Uni_db.courses to 'Professor'@'localhost';
grant select, update on Uni_db.grades to 'Professor'@'localhost';

-- Student
grant select on Uni_db.grades to 'Student'@'localhost';
grant select on Uni_db.courses to 'Student'@'localhost';

-- Registrar
grant all privileges on Uni_db.students to 'Registrar'@'localhost';
grant all privileges on Uni_db.courses to 'Registrar'@'localhost';


Show Grants for 'Professor'@'localhost';
Show grants for 'Student'@'localhost';
Show grants for 'Registrar'@'localhost';


Drop database Uni_db;
Drop table courses;
Drop table students;
Drop table grades;

Drop user Professor;
Drop user Student;
Drop user Registrar;
