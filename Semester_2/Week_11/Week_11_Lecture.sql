Create Database MYDB;
Use MYDB;

Create table Customers(
Customer_ID INT,
Customer_Name Varchar(50),
City varchar(50)
);

Insert Into Customers Values
(1, "Friar Tuck", "Nottingham"),
(2, "Jebidiah Kerman", "Southampton"),
(3, "Valentina Kerman", "Winchester"),
(4, "Bob Kerman", "Florida");

 
 Create Table Orders(
 Order_ID Int,
 Customer_ID Int,
 Product Varchar(50)
 );
 
 Insert Into Orders Values
 (101, 1, "Gold"),
 (102, 3, "Parts"),
 (103, 1, "Stuff"),
 (104, 5, "Laptop");
 
 
 -- INNER JOIN
SELECT *
FROM Customers 
INNER JOIN Orders ON Customers.Customer_ID = Orders.Customer_ID;

-- Left join
Select *
From Customers 
Left Join Orders ON Customers.Customer_ID = Orders.Customer_ID;

-- Right Join
Select *
From Customers 
Right Join Orders ON Customers.Customer_ID = Orders.Customer_ID;

-- Full Outer Join
-- It is only supported by orcacle and not in MySQL
Select *
From Customers Left Join Orders
ON Customers.Customer_ID = Orders.Customer_ID
Union
Select *
From Customers Right Join Orders
ON Customers.Customer_ID = Orders.Customer_ID;

-- Cross Join
Select *
FROM Customers
Cross Join Orders;

 