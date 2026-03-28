const express = require('express');
const mongoose = require('mongoose');
const feedRoutes = require('./routes/feed');
const app = express();
app.use(express.json()); // Allows the server to understand JSON sent from the frontend

app.use(express.static('public')); // Serves our HTML testing page automatically
app.use('/feed', feedRoutes); // Routes all blog-related requests to the feed file
// Replace this with YOUR actual connection string from Step 1
const MONGODB_URI = 'mongodb+srv://jhetherton24_db_user:z7P21BOrHr5iOpRv@cluster0.1luioxg.mongodb.net/?appName=Cluster0';
mongoose.connect(MONGODB_URI)
    .then(() => {
        app.listen(3000); // Only start the server if the database connection works
        console.log('Success! Database connected and server running on port 3000');
})
.catch(err => console.log('Database connection error:', err));