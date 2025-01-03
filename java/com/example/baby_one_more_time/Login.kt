package com.example.baby_one_more_time
// 아웃풋을 만든다. 서버에서 호출했을 때 받아오는 응답값
data class Login(
    var code : String,
    var msg : String
)