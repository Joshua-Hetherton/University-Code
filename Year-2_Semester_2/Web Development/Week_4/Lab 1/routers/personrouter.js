const { Router } = require('express'); const
app = Router();
app.get('/person/:personId', (req, res) => {
res.send(req.params);
});
module.exports = app;