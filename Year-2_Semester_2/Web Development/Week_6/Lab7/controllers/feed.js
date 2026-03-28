const Post = require('../models/post'); // Import our blueprint/model
// 1. READ Operation: Get all posts
exports.getPosts = (req, res, next) => {
    Post.find()
    .then(foundPosts => {
        res.status(200).json({ posts: foundPosts });
    })
    .catch(err => res.status(500).json({ error: 'Failed to fetch' }));
};

// 2. CREATE Operation: Save a new post
exports.createPost = (req, res, next) => {
    const post = new Post({ // Create a new instance based on our model
        title: req.body.title, // Take data from the frontend form
        content: req.body.content
    });

    post.save() // This actually sends the data to MongoDB Atlas
    .then(result => {
        res.status(201).json({ message: 'Saved!', post: result });
    })
    .catch(err => res.status(500).json({ error: 'Failed to save' }));
};

// 3. UPDATE Operation: Edit a specific post
exports.updatePost = (req, res, next) => {  
    const postId = req.params.postId; // Get the unique ID from the URL

    Post.findById(postId) // Find the exact document first
    .then(post => {
        if (!post) { return res.status(404).json({ message: 'Not found' }); }
        post.title = req.body.title; // Replace old data with new data
        post.content = req.body.content;
        return post.save(); // Save the modified document
    })
    .then(result => res.status(200).json({ message: 'Updated!', post: result }))
    .catch(err => res.status(500).json({ error: 'Update failed' }));
};

// 4. DELETE Operation: Remove a post
exports.deletePost = (req, res, next) => {
    const postId = req.params.postId;
    
    Post.findByIdAndDelete(postId) // Find it and remove it in one step
    .then(() => res.status(200).json({ message: 'Deleted' }))
    .catch(err => res.status(500).json({ error: 'Delete failed' }));
};