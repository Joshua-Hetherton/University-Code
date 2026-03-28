const mongoose = require('mongoose'); // Import the mongoose library
const Schema = mongoose.Schema; // Get the Schema constructor
// This blueprint ensures every 'Post' has a title and content
const postSchema = new Schema({
    title: {
        type: String,
        required: true // The database will reject a post if the title is missing
    },
    content: {
            type: String,
            required: true // Ensures we don't save empty posts
    }
});
// We 'model' the schema into a tool we can use to perform queries
module.exports = mongoose.model('Post', postSchema);