package com.example.baby_one_more_time

data class UserSignupResponse(
    val id: Int, // 생성된 사용자 ID
    val name: String,
    val phone_number: String,
    val gender: String,
    val Birthday: String,
    val Email: String,
    val Password: String,
    val created: String // 생성 시간
)
