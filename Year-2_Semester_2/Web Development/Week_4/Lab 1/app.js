const express= require('express');

const home=require('./routers/homerouter');
const about=require('./routers/aboutouter');
const contact=require('./routers/contactrouter');
const books=require('./routers/booksrouter');
const person=require('./routers/personrouter');


const app= express();

app.use(home);
app.use(about);
app.use(contact);
app.use(books);
app.use(person);

module.exports = app;

// const express = require('express'); 
// const app = express();
// app.get('/', (req, res) => {
// res.send('Hello World!');
// });
// module.exports = app;