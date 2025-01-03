package com.example.baby_one_more_time

import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.POST

interface SignupService {
    @POST("userlist/") // Django `user_list` POST 엔드포인트
    fun signupUser(@Body user: UserSignupRequest): Call<UserSignupResponse>
}