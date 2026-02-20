const express=require("express");

const app=express();

const chalk=require("chalk");

const path=require("path");

const debug = require("debug")("app");

const morgan = require("morgan");
app.use(morgan("tiny"));

app.get("/",(req,res)=>{
    res.sendFile(path.join(__dirname, "views/index.html"));
});

app.listen(3000,()=>{
    debug(`Server is running on port ${chalk.green("3000")}`);
});

//node app.js
//set DEBUG=* && node app.js