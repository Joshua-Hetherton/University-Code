Create database ecommercedb;
use ecommercedb;

create table orders(

orderid int Primary Key auto_increment,
customer_name varchar(50),
product varchar(50),
category varchar(50),
price decimal(10,2),
quantity int,
order_date date
);
drop table orders;

INSERT INTO orders (customer_name, product, category, price, quantity, order_date) VALUES
("Bob", "GPU sag Bracket", "Accessories", 5.95, 1, '2025-04-01'),
("Grace", "Free Iphone 16", "Phone", 0.01, 1, '2025-04-02'),
("Valentina Kerman", "Rocket Parts", "Materials", 100.00, 30, '2025-04-03'),
("Van von Kerman", "SLS System", "Safety", 100.00, 30, '2025-04-03'),
("Jebediah Kerman", "Reaction Wheels", "Stability", 250.00, 4, '2025-04-04'),
("Bill Kerman", "Struts", "Structural", 15.00, 10, '2025-04-05'),
("Bob Kerman", "Solar Panels", "Power", 120.00, 6, '2025-04-06'),
("Wernher von Kerman", "Advanced Graviton Detector", "Science", 9999.99, 1, '2025-04-07'),
("Bruce", "Galleon", "Piracy", 1500.00, 1, '2025-04-08'),
("Jeff", "Moon Base Kit", "Habitation", 500000.00, 1, '2025-04-09'),
("Tim C.", "Apple Vision Ultra Pro Max", "Wearable Tech", 3499.99, 1, '2025-04-10'),
("Valentina Kerman", "Parachutes", "Safety", 50.00, 5, '2025-04-11'),
("Gene Kerman", "Mission Control Desk", "Furniture", 1200.00, 1, '2025-04-12'),
("Timmy", "6x Weetabix", "Food", 10.00, 1, '2025-04-13'),
("Linus", "R&D Blueprints", "Research", 500.00, 3, '2025-04-14'),
("Gus Kerman", "SRBs", "Propulsion", 250.00, 8, '2025-04-15'),
("Man Like Rebecca", "VSCodeKiller", "Malware", 1.00, 1, '2025-04-16'),
("Chris Hadfield", "Space Guitar", "Entertainment", 299.99, 1, '2025-04-17'),
("Buzz Aldrin", "Moon Rock", "Collectibles", 99999.99, 1, '2025-04-18'),
("Neil Armstrong", "Apollo 11 Flag Replica", "Memorabilia", 75.00, 2, '2025-04-19');

-- Task 3
-- Count Order
Select Count(*) As total_orders
From orders;
-- Calculate Total Revenue
Select SUM(price) AS total_revenue
From orders;
-- Determine total of products sold
Select SUM(quantity) AS total_products
From orders;
-- Calc avg of all product
Select AVG(price) AS avg_price
From orders;

-- Task 4
-- Convert customer names to uppercase
Select customer_name, Upper(customer_name) AS upper_name
From orders;
-- Extract first 3 characters from each product
select left(product, 3) as product_prefix
from orders;
-- Display the length of each customers name
Select customer_name, Length(customer_name) As name_length
From orders;
-- Round all product prices to nearest whole number
Select price, round(price, 0) AS rounded_price
From orders;
-- Display the current data and time from the system
SELECT NOW() AS current_time_val;
-- Sort all customer names in alphabetical order after converting to uppercase
SELECT customer_name FROM orders
Order By customer_name asc;
