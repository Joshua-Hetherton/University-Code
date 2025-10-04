Create Database Tournament;
Use Tournament;

Create Table Players(
PlayerID INT Primary Key,
GamerTag Varchar(50),
Country Varchar(50)
);
Drop Table Players;
Create Table Matches(
MatchID Int Primary Key,
PlayerID Int,
Game varchar(50)
);

Insert Into Players Values
(1, "A", "UK"),
(2, "B", "Switzerland"),
(3, "C", "Germany"),
(6, "D", "France"),
(7, "E", "US"),
(5,"F", "Canada");

Insert Into Matches Values
(1, 2, "Super Smash Bros"),
(2, 8, "GTA V"),
(3, 10, "Sea of Thieves"),
(4, 3, "Balatro"),
(5, 7, "Noita" ),
(8, 6, "Factorio"),
(10, 9, "Stardew Valley"),
(6, 5, "Kerbal Space Program");


-- Inner Join
Select *
From Players
Inner Join Matches On Players.PlayerID = Matches.PlayerID;

-- Left Join
Select *
From Players
Left Join Matches On Players.PlayerID = Matches.PlayerID;

-- Right Join
Select *
From Players
Right Join Matches on Players.PlayerID= Matches.PlayerID;

-- Full Outer Join
Select *
From Players
Left Join Matches On Players.PlayerID = Matches.PlayerID
Union
Select *
From Players
Right Join Matches on Players.PlayerID= Matches.PlayerID;

-- Task 4
-- No Registered Person
SELECT Game
FROM Matches
WHERE PlayerID NOT IN (SELECT PlayerID FROM Players)
OR PlayerID IS NULL;

-- Listing each player and No. of matches played
SELECT Players.PlayerID, Players.GamerTag, COUNT(Matches.MatchID) AS MatchesPlayed
FROM Players
LEFT JOIN Matches ON Players.PlayerID = Matches.PlayerID
GROUP BY Players.PlayerID, Players.GamerTag;
