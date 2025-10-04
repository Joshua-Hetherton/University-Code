Create Database Sample;
use Sample;

Create Table TableA(
ID INT,
Val Varchar(50)
);
 
 Insert Into TableA Values
 (1, "Fox"),
 (2, "Cop"),
 (3, "Taxi"),
 (6, "Washington"),
 (7, "Dell"),
 (5, "Arizona"),
 (4, "Lincoln"),
 (10, "Lucent");
 
 Create Table TableB(
 ID INT,
 Val Varchar(50)
 );
 
 Insert Into TableB Values
 (1, "Trot"),
 (2, "Car"),
 (3, "Cab"),
 (6, "Monument"),
 (7, "PC"),
 (8, "Microsoft"),
 (9, "Apple"),
 (11, "Scotch");
 
 
  -- INNER JOIN
SELECT *
FROM TableA
INNER JOIN TableB ON TableA.ID = TableB.ID;

-- Left join
Select *
From TableA
Left Join TableB ON TableA.ID = TableB.ID;

-- Right Join
Select *
From TableA
Right Join TableB ON TableA.ID = TableB.ID;


-- Full Outer Join
-- It is only supported by orcacle and not in MySQL
Select *
From TableA
Left Join TableB ON TableA.ID = TableB.ID
Union
Select *
From TableA
Right Join TableB ON TableA.ID = TableB.ID;


-- Cross Join
Select *
FROM TableA
Cross Join TableB;

 
 
 