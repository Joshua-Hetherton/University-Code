const express = require('express');
const app = express();
app.use(express.static('public')); //tells Express to serve everything in the 'public' folder
app.listen(3000, () => {
console.log('Server is running! Visit http://localhost:3000');
});