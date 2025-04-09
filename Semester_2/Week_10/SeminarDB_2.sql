Create database librarydb;
use librarydb;

create table books (
book_id int primary key auto_increment,
title varchar(100),
author varchar(50),
genre varchar(30),
price decimal(10,2),
copies int,
publish_date date

);

INSERT INTO books(title, author, genre, price, copies, publish_date) VALUES
("Red Rising", "Pierce Brown", "Science-Fiction", 12.99, 50, '2014-01-28'),
("Golden Son (Red Rising Trilogy)", "Pierce Brown", "Science-Fiction", 13.99, 45, '2015-01-06'),
("Morning Star (Red Rising Trilogy)", "Pierce Brown", "Science-Fiction", 14.99, 40, '2016-02-09'),
("Iron Gold (Red Rising Saga)", "Pierce Brown", "Science-Fiction", 16.99, 35, '2018-01-16'),
("The Way of Kings (Stormlight Archive)", "Brandon Sanderson", "Fantasy", 14.99, 30, '2010-08-31'),
("Words of Radiance (Stormlight Archive)", "Brandon Sanderson", "Fantasy", 16.99, 25, '2014-03-04'),
("Oathbringer (Stormlight Archive)", "Brandon Sanderson", "Fantasy", 18.99, 20, '2017-11-14'),
("Rhythm of War (Stormlight Archive)", "Brandon Sanderson", "Fantasy", 20.99, 15, '2020-11-17'),
("Throne of Glass", "Sarah J. Maas", "Fantasy", 12.99, 50, '2012-08-02'),
("Crown of Midnight (Throne of Glass)", "Sarah J. Maas", "Fantasy", 13.99, 45, '2013-08-27'),
("Heir of Fire (Throne of Glass)", "Sarah J. Maas", "Fantasy", 14.99, 40, '2014-09-02'),
("Queen of Shadows (Throne of Glass)", "Sarah J. Maas", "Fantasy", 15.99, 35, '2015-09-01'),
("Empire of Storms (Throne of Glass)", "Sarah J. Maas", "Fantasy", 16.99, 30, '2016-09-06'),
("Tower of Dawn (Throne of Glass)", "Sarah J. Maas", "Fantasy", 17.99, 25, '2017-09-05'),
("Kingdom of Ash (Throne of Glass)", "Sarah J. Maas", "Fantasy", 18.99, 20, '2018-10-23'),
("A Court of Thorns and Roses", "Sarah J. Maas", "Fantasy", 14.99, 30, '2015-05-05'),
("A Court of Mist and Fury", "Sarah J. Maas", "Fantasy", 15.99, 28, '2016-05-03'),
("A Court of Wings and Ruin", "Sarah J. Maas", "Fantasy", 16.99, 25, '2017-05-02'),
("A Court of Frost and Starlight", "Sarah J. Maas", "Fantasy", 17.99, 20, '2018-05-01'),
("A Court of Silver Flames", "Sarah J. Maas", "Fantasy", 18.99, 18, '2021-02-16');

-- Task 3
-- Count total number of books
Select count(*) as total_books
From books;
-- Find the most and least expensive books
Select min(price) as cheapest
from books;
Select max(price) as expensive
from books;
-- Calculate the average book price
select AVG(price) as avg_salary
from books;
-- Group the Books by genre and count how many in each
Select genre AS book_genre from books GROUP by genre;
-- display only genres with more than 5
Select genre AS book_genre from books GROUP by genre
having Count(*)>5;

-- Task 4
-- Convert all book titles to uppercase
Select title, Upper(title) AS title_upper
From books;
-- Extract the first five characters of each authors name
select left(author, 5) as auth_prefix
from books;
-- Find the lengths of each books title
select title, length(title) as title_len
From books;
-- Round all book prices to 2d.p, USELESS
Select price, round(price, 2) AS rounded_price
From books;
-- Display the current date and time
SELECT NOW() AS current_time_val;
-- Sort Book titles in ascending order
SELECT title FROM books
Order By title asc;