package com.example.baby_one_more_time

import retrofit2.Call
import retrofit2.http.Field
import retrofit2.http.FormUrlEncoded
import retrofit2.http.POST

interface LoginService {
    @FormUrlEncoded
    @POST("/app_login/")
    fun requestLogin(
        // 인풋을 정의하는 곳
        @Field("userid") userid : String,
        @Field("userpw") userpw : String
    ) : Call<Login> // 아웃풋을 정의하는 곳
}