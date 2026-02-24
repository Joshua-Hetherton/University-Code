const express=require("express");

const app=express();

const chalk=require("chalk");

const path=require("path");

const debug = require("debug")("app");

const morgan = require("morgan");
app.use(morgan("tiny"));

const blogrouter=require("./src/routes/blogroutes");
app.use("/blogs", blogrouter);

app.set("views", "./src/views");
app.set("view engine", "ejs");

app.use(express.static(path.join(__dirname, "/public")));
app.use("/css", express.static(path.join(__dirname, "/public/css")));
app.use("/js", express.static(path.join(__dirname, "/public/js")));

app.get("/",(req,res)=>{
    res.render("index", {title: "My Homepage"});
});

app.listen(3000,()=>{
    debug(`Server is running on port ${chalk.green("3000")}`);
});

//node app.js
//set DEBUG=* && node app.js
//npm start
//On rendering additional pages