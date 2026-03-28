const express = require('express');
const feedController = require('../controllers/feed'); // Import the "brain" controller
const router = express.Router(); // Create the router object

router.get('/posts', feedController.getPosts); // GET /feed/posts

router.post('/post', feedController.createPost); // POST /feed/post

router.put('/post/:postId', feedController.updatePost); // PUT /feed/post/:postId

router.delete('/post/:postId', feedController.deletePost); // DELETE /feed/post/:postId

module.exports = router; // We must export the router so app.js can use it.