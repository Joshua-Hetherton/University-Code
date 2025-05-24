-- Create the database and switch to it
CREATE DATABASE IF NOT EXISTS Bookstore;
USE Bookstore;

-- Create Orders table
CREATE TABLE Orders (
	order_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    book_title VARCHAR(100)
);

-- Create Payments table with a foreign key to Orders
CREATE TABLE Payments (
	payment_id INT PRIMARY KEY,
    order_id INT,
    amount DECIMAL(10,2),
    FOREIGN KEY (order_id) REFERENCES Orders(order_id)
);

-- Start a transaction
START TRANSACTION;

-- Insert a valid order
INSERT INTO Orders (order_id, customer_name, book_title)
VALUES (1, 'Ravi', 'The Alchemist');

-- Create a savepoint after inserting the order
SAVEPOINT sp_after_order;

-- Insert a payment with an invalid order_id to simulate error
-- This will fail due to foreign key constraint
-- but we proceed as if it were a logical error instead
-- so assume it executes but is incorrect
-- then rollback to remove it
INSERT INTO Payments (payment_id, order_id, amount)
VALUES (101, 999, 500.00);

-- Rollback to the savepoint (removes faulty payment insert)
ROLLBACK TO sp_after_order;

-- Insert correct payment
INSERT INTO Payments (payment_id, order_id, amount)
VALUES (102, 1, 499.99);

-- Commit the final valid state
COMMIT;

-- Show tables and content
SHOW TABLES;
SELECT * FROM Orders;
SELECT * FROM Payments;

-- Clean up
DROP TABLE Payments;
DROP TABLE Orders;
