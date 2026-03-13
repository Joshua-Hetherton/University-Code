// This array lives in the server's RAM.
// It will reset every time you restart the server.

let posts= [
    {id: 1, title: "Welcome!", content:"This is a hardcoded post from the server."}
];

// GET: Send the current list of posts to the requester
exports.getPosts = (req, res) => {
    res.status(200).json({posts:posts});
};

// POST: Take data from the request body and add it to our list
exports.createPost = (req, res) => {   
    const newPost={
        id: new Date().toISOString(), // Temporary ID logic
        title: req.body.title,
        content: req.body.content
    };
    posts.push(newPost);
    res.status(201).json({message: "Post added to server memory", post:newPost});
};


    