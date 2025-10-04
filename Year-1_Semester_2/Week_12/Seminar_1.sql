Create database Hospital;
use Hospital;

-- Patient details
Create table patients(
	patient_id Int primary Key,
    name varchar(100),
    dob Date,
    diagnosis Text
);

Create table appointments(
	appointment_id int Primary key,
    patient_id int,
    doctor_name varchar(100),
    appointment_date Date
);

Create user 'reception'@'localhost' identified by 'recep@123';

Create user 'doctor1'@'localhost' identified by 'doc@123';

Create user 'adminuser'@'localhost' identified by 'admin@123';

Grant Select, insert on Hospital.appointments To 'reception'@'localhost';

Grant Select on Hospital.patients To 'doctor1'@'localhost';
Grant Select on Hospital.appointments To 'doctor1'@'localhost';

Grant All privileges on hospital.* To 'adminuser'@'localhost';

Revoke Select on Hospital.patients From 'doctor1'@'localhost';

Show Grants for 'reception'@'localhost';
Show grants for 'doctor1'@'localhost';
Show grants for 'adminuser'@'localhost';


-- Deleting stuff
DROP USER 'reception'@'localhost';
DROP USER 'doctor1'@'localhost';
DROP USER 'adminuser'@'localhost';
drop database Hospital;
