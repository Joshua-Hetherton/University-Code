const { Router } = require('express'); const
app = Router();
app.get('/books', (req, res) => {
res.send("Books");
});
module.exports = app;