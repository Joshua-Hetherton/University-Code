const express= require ("express");
const app= express();

// Middleware to parse incoming JSON data from the frontend
app.use(express.json());

// Serve static files (this will host our HTML page)
app.use(express.static("public"));

//Adding feed routes
const feedRoutes= require("./routes/feed");
app.use("/feed", feedRoutes);

app.listen(3000, () => {
    console.log("Server is running on http://localhost:3000");
});

