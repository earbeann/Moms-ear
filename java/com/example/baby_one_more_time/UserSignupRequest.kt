package com.example.baby_one_more_time

data class UserSignupRequest(
    val name: String,
    val gender: String,
    val phone_number: String,
    val Birthday: String, // Django DateField 포맷에 맞게 전달
    val Email: String,
    val Password: String
)
