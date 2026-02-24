const express = require("express");

const blogrouter=express.Router();  

blogrouter.get("/",(req,res)=>{
    res.render("blogs");
});

module.exports=blogrouter;