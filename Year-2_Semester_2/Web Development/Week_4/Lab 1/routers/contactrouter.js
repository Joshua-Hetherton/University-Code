const { Router } = require('express'); const
app = Router();
app.get('/contact', (req, res) => {
res.send("Contact");
});
module.exports = app;