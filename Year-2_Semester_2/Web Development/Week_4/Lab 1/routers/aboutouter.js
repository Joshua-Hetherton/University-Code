const { Router } = require('express'); const
app = Router();
app.get('/about', (req, res) => {
res.send("About");
});
module.exports = app;