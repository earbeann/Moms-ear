package com.example.baby_one_more_time.network

// ApiService.kt
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.POST

// 요청 데이터 클래스
data class MusicRequest(val text_prompt: String)

// 응답 데이터 클래스
data class MusicResponse(val music_url: String)

// Retrofit 인터페이스
interface ApiService {
    @POST("api/generate_music/")
    fun generateMusic(@Body request: MusicRequest): Call<MusicResponse>
}
